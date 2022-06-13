#!/usr/bin/python
import argparse
# import os
import random
import time

import face_recognition
# import dlib
import cv2
import imutils
import numpy as np

from face_recognition_new import detect_landmarks


biden_image = face_recognition.load_image_file("Janna2.jpg")
janna_face_encoding = face_recognition.face_encodings(biden_image)[0]

# Создание массивов со всеми людьми и добавление им имен
known_face_encodings = [
    janna_face_encoding,
]
known_face_names = [
    "Janna",
]

face_locations = []
face_encodings = []
face_names = []


# Проверка, находится ли точка внутри прямоугольника
def rect_contains(rectangle, point):
    if point[0] < rectangle[0]:
        return False
    elif point[1] < rectangle[1]:
        return False
    elif point[0] > rectangle[2]:
        return False
    elif point[1] > rectangle[3]:
        return False
    return True


# Рисуем точку
def draw_point(img, p, color):
    cv2.circle(img, p, 2, color, cv2.FILLED, cv2.LINE_AA, 0)


# Рисуем треугольники Делоне
def draw_delaunay(img, subdiv, delaunay_color):
    triangle_list = subdiv.getTriangleList()
    size = img.shape
    r = (0, 0, size[1], size[0])

    for t in triangle_list:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rect_contains(r, pt1) and rect_contains(r, pt2) and rect_contains(r, pt3):
            cv2.line(img, pt1, pt2, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt2, pt3, delaunay_color, 1, cv2.LINE_AA, 0)
            cv2.line(img, pt3, pt1, delaunay_color, 1, cv2.LINE_AA, 0)


def delaunay_triangulation(img, points, process_this_frame=True):
    global face_encodings, face_locations, face_names

    # Увеличиваем размер кадра видео до 1/4 для более быстрой обработки распознавания лиц.
    small_frame = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)

    # Преобразуйте изображение из цвета BGR (который использует OpenCV) в цвет RGB (который использует face_recognition).
    rgb_small_frame = small_frame[:, :, ::-1]

    # Обрабатывайте только каждый второй кадр видео, чтобы сэкономить время
    if process_this_frame:
        # Находит все лица и кодировки лиц в текущем кадре видео
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Проверка, совпадает ли лицо с известным лицом (лицами)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "None"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)


    # Отображение результатов
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Изменения масштаба расположение лиц, поскольку обнаруженный нами кадр был увеличен до размера 1/4
        top *= 4
        right *= 4
        bottom += 54
        left *= 4

        # Отрисовка имени в прямоугольника на видео
        cv2.rectangle(img, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


    # Определение цветов для рисования
    delaunay_color = (255, 255, 255)

    # Прямоугольник для использования с Subdiv2D
    size = img.shape
    rect = (0, 0, size[1], size[0])

    # Создание экземпляра Subdiv2D
    subdiv = cv2.Subdiv2D(rect)

    # Вставка точек внутри экземпляра
    for p in points:
        subdiv.insert(p)

    # Рисуем треугольники Делоне
    draw_delaunay(img, subdiv, delaunay_color)

    # Выводим результат на экран
    cv2.imshow('Video', img)


    # Добавляется бесконечная задержка для работы камеры не прирывно
    cv2.waitKey(1)

    return img


def main():
    video_capture = cv2.VideoCapture(0) # Подключаем камеру
    while True:
        ret, img = video_capture.read() # Получаем один кадр видео

        # Изменения размера изображения для более быстрой обработки видео
        img = imutils.resize(img, width=800)

        # Обнаружение ориентиров с помощью модели
        _, landmarks = detect_landmarks(img)


        # Если l28 использует только 28 из 68 ориентиров, предоставляемых dlib shape-predictor
        l28 = args.l28
        if l28:
            mask = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 21, 22, 23, 25, 27, 29, 30, 31, 35, 36, 39, 42, 45, 48, 51,
                    54, 57]
            landmarks = [landmarks[i] for i in mask]

        # Рисуем на экране линии
        delaunay_triangulation(img, landmarks)


if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='Process image for facial recognition analysis and visualization.')
    parser.add_argument('--voronoi', action='store_true', help='show voronoi diagrams of recognized face',
                        required=False)
    parser.add_argument('--l28', action='store_true', help='only use 28 landmarks', required=False)
    args = parser.parse_args()

    # Call main function
    main()
