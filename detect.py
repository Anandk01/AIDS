            if center:
                cx, cy = center
                cv2.circle(frame, center, 10, (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, f"Tracking ({cx},{cy})", (cx + 10, cy - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

                servo_x, servo_y = map_position_to_servo(cx, cy, WIDTH, HEIGHT)
                servo_controller.update_position(servo_x, servo_y)

            else:
                cv2.putText(frame, "No Human Detected", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Body Tracker", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pose_tracker.release()


if __name__ == "__main__":
    main()
