
import time
import cv2
import mediapipe as mp
import open3d as o3d
import numpy as np

MODEL_PATH = "hand_landmarker.task"

# Hand skeleton connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def create_shape(shape_name):
    """Create a 3D shape centered around its origin."""
    if shape_name == "sphere":
        mesh = o3d.geometry.TriangleMesh.create_sphere(
            radius=0.15,
            resolution=24
        )
        mesh.paint_uniform_color([0.1, 0.5, 1.0])
    else:
        mesh = o3d.geometry.TriangleMesh.create_box(
            width=0.3,
            height=0.3,
            depth=0.3
        )
        mesh.translate([-0.15, -0.15, -0.15])
        mesh.paint_uniform_color([1.0, 0.35, 0.1])

    mesh.compute_vertex_normals()
    return mesh


def fingers_extended(hand):
    """Return whether the four fingers are extended."""
    # For a mirrored webcam view, tip y above PIP y means extended.
    pairs = [(8, 6), (12, 10), (16, 14), (20, 18)]
    return all(hand[tip].y < hand[pip].y for tip, pip in pairs)


def main():
    if not __import__("os").path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Download hand_landmarker.task and put it beside main.py"
        )

    # Create Open3D viewer
    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name="Gesture3D - Move Objects With Your Hand",
        width=800,
        height=600
    )

    shape_name = "sphere"
    mesh = create_shape(shape_name)
    viewer.add_geometry(mesh)

    # Save the original vertices so scaling does not accumulate.
    base_vertices = np.asarray(mesh.vertices).copy()

    # Object position and scale
    position = np.array([0.0, 0.0, 0.0])
    scale = 1.0

    # Smooth movement
    smoothing = 0.25
    previous_open_palm = False

    # Setup MediaPipe Tasks API
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1
    )

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        viewer.destroy_window()
        raise RuntimeError(
            "Webcam could not open. Check camera permissions."
        )

    start_time = time.monotonic()
    last_timestamp = -1

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(
            options
        ) as detector:

            while True:
                ok, frame = camera.read()

                if not ok:
                    print("Could not read webcam frame.")
                    break

                # Mirror webcam image
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                # Convert BGR to RGB for MediaPipe
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                # VIDEO mode requires increasing timestamps
                timestamp = int(
                    (time.monotonic() - start_time) * 1000
                )
                timestamp = max(timestamp, last_timestamp + 1)
                last_timestamp = timestamp

                result = detector.detect_for_video(
                    image, timestamp
                )

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    # Draw landmarks and skeleton
                    points = []
                    for landmark in hand:
                        px = int(landmark.x * width)
                        py = int(landmark.y * height)
                        points.append((px, py))

                    for a, b in CONNECTIONS:
                        cv2.line(
                            frame, points[a], points[b],
                            (255, 180, 0), 2
                        )

                    for point in points:
                        cv2.circle(
                            frame, point, 4,
                            (0, 255, 0), -1
                        )

                    # 1. Move object with index fingertip (landmark 8)
                    index_tip = hand[8]

                    target = np.array([
                        (0.5 - index_tip.x) * 2.0,
                        (0.5 - index_tip.y) * 1.5,
                        0.0
                    ])

                    position = (
                        (1 - smoothing) * position
                        + smoothing * target
                    )

                    # 2. Resize object based on thumb-index pinch
                    thumb_tip = hand[4]
                    distance = np.hypot(
                        thumb_tip.x - index_tip.x,
                        thumb_tip.y - index_tip.y
                    )

                    # Close fingers = smaller; spread fingers = larger
                    scale = float(np.clip(
                        (distance - 0.025) / (0.16 - 0.025),
                        0.5,
                        1.8
                    ))

                    # 3. Open palm switches sphere <-> cube
                    open_palm = fingers_extended(hand)

                    if open_palm and not previous_open_palm:
                        viewer.remove_geometry(
                            mesh, reset_bounding_box=False
                        )

                        shape_name = (
                            "cube" if shape_name == "sphere"
                            else "sphere"
                        )

                        mesh = create_shape(shape_name)
                        base_vertices = np.asarray(
                            mesh.vertices
                        ).copy()

                        viewer.add_geometry(
                            mesh, reset_bounding_box=False
                        )

                    previous_open_palm = open_palm

                    cv2.putText(
                        frame, f"Shape: {shape_name}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2
                    )
                    cv2.putText(
                        frame, f"Pinch distance: {distance:.3f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2
                    )

                else:
                    previous_open_palm = False
                    cv2.putText(
                        frame, "Show your hand",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2
                    )

                # Rebuild the mesh from original vertices every frame
                new_vertices = base_vertices * scale + position
                mesh.vertices = o3d.utility.Vector3dVector(
                    new_vertices
                )
                mesh.compute_vertex_normals()

                # Tell Open3D to redraw the changed geometry
                viewer.update_geometry(mesh)
                viewer.poll_events()
                viewer.update_renderer()

                cv2.putText(
                    frame,
                    "Move index | Pinch to resize | Open hand to switch",
                    (10, height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1
                )

                cv2.imshow("Gesture3D Webcam", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        viewer.destroy_window()


if __name__ == "__main__":
    main()
