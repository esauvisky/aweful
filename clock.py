
    # if routine == "clock":
    #     index = 0
    #     # get the latest used index number
    #     for group in os.listdir("./data"):
    #         for filename in os.listdir("./data/" + group):
    #             if filename.endswith(".jpg"):
    #                 num = int(filename.split(".")[0].split("-")[0])
    #                 if num >= index:
    #                     index = num + 1

    #     sleep_counter = deque(maxlen=6 * 60)
    #     images = deque(maxlen=SEQUENCE_LENGTH)
    #     path = f"./data/{int(group) + 1}/"

    #     while True:
    #         subprocess.run(shlex.split(f"fswebcam /tmp/aweful_tmp.jpg -d /dev/video0 -S2 -F1"),
    #                        check=False,
    #                        stdout=subprocess.DEVNULL,
    #                        stderr=subprocess.DEVNULL)

    #         image = get_image(f"/tmp/aweful_tmp.jpg", IMAGE_HEIGHT, IMAGE_WIDTH)
    #         images.append(image)
    #         # # image = image_utils.load_img(f"{path}{index}.jpg", color_mode="grayscale")
    #         # # image = image_utils.img_to_array(image)
    #         # # image = image[:-21, :]

    #         # resized = image_utils.array_to_img(image).resize((IMAGE_WIDTH, IMAGE_HEIGHT))
    #         # resized = image_utils.img_to_array(resized, dtype=np.float32) / 255.0
    #         # images.append(resized)

    #         if len(images) < SEQUENCE_LENGTH:
    #             # sleep_counter.append(0)
    #             # logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
    #             continue

    #         X_loop = np.array([images], dtype=np.float32)
    #         y_loop = model.predict(X_loop, verbose=0)
    #         y_loop_class = np.round(y_loop).flatten().astype(int)[0]

    #         date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #         prediction = "Awake 🆙" if y_loop_class == 0 else "Sleep 💤"

    #         if y_loop_class == 1:
    #             sleep_counter.append(1)
    #             logger.warning(f"Sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")
    #         else:
    #             sleep_counter.append(0)
    #             logger.info(f"No sleep detected {sleep_counter.count(1)} / {len(sleep_counter)}")

    #         index += 1
    #         # copy the image to the data folder
    #         shutil.copy("/tmp/aweful_tmp.jpg", f'{path}{index}-{"awake" if y_loop_class == 0 else "sleep"}.jpg')

    #         if sleep_counter.count(1) > 0.8 * 6 * 60:
    #             print("Wake up! You've been sleeping for more than 6 hours!")
    #             alarm_triggered = True
    #         else:
    #             alarm_triggered = False
