import cv2
import dlib
import imutils


def detect_landmarks(img):
    # Устанавливаем 68-точечный детектор лицевых ориентиров
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # преобразование в детектор оттенков серого
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # обнаружение лиц на изображении
    faces_in_image = detector(img_gray, 0)

    if faces_in_image:

        # Обработка только первое изображение
        face = faces_in_image[0]

        # Назначение лицевых точек
        landmarks = predictor(img_gray, face)

        # Распаковка 68 координат ориентиров из объекта dlib в список
        landmarks_list = []
        for i in range(0, landmarks.num_parts):
            landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

        return face, landmarks_list
    return "", ""


if __name__ == '__main__':
    # Устанавливаем 68-точечный детектор лицевых ориентиров
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    video_capture = cv2.VideoCapture(0) # Запуск камеры
    while True:
        ret, img = video_capture.read()

        # Уменьшение размера фото для более быстрой обработки
        img = imutils.resize(img, width=800)

        # преобразование в оттенки серого
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # обнаружение лиц на изображении
        faces_in_image = detector(img_gray, 0)

        # Проходим циклом по каждой точки изображения
        for face in faces_in_image:

            # Поиск точек на лице
            landmarks = predictor(img_gray, face)

            # Распаковываем 68 координат ориентиров точек из объекта dlib в список
            landmarks_list = []
            for i in range(0, landmarks.num_parts):
                landmarks_list.append((landmarks.part(i).x, landmarks.part(i).y))

            # Для каждого ориентира наносится график и пишется его номер
            for landmark_num, xy in enumerate(landmarks_list, start=1):
                cv2.circle(img, (xy[0], xy[1]), 12, (168, 0, 20), -1)
                cv2.putText(img, str(landmark_num), (xy[0] - 7, xy[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255),
                            1)

        # Отображение результата на экран
        cv2.imshow('img', img)
        cv2.waitKey(1)
