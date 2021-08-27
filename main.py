import math
import pygame as pg
from math import sin, cos, floor, ceil, log
from colorsys import hsv_to_rgb
from numba import njit
import time
from functools import cache
import numpy as np
from PIL import Image, ImageDraw
import os
from threading import Thread
from pprint import pprint
from queue import LifoQueue, Empty

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIME = (0, 255, 0)
# END COLORS


FPS = 30
size = (600, 600)
scale = 1
zoom = 1
scaleZoomKoeficient = 1.1
position = (0, 0)
RES = 2
mandelBrotCache = {}
mandelBrotQueue = LifoQueue()
imageQueue = LifoQueue()
bufferQueueIn = LifoQueue()
bufferQueueOut = LifoQueue()
positionSnap = 1
running = True
lastImageBuffer = (Image.new("RGB", size), (0, 0))
# imageTimes = [time.time()]

pg.init()
display = pg.display.set_mode(size, pg.HWSURFACE | pg.DOUBLEBUF)

if not os.path.exists("renderedSets"):
    os.mkdir("renderedSets")


def lerp(point1, point2, t):
    return tuple(
        point1[i] + (point2[i]-point1[i])*t for i in range(len(point1))
    )


class Mouse:
    draging = False
    startPos = (0, 0)
    lastDragFrame = 0


class ImageBuffer:
    def __init__(self):
        self.images: dict[int, Image.Image] = {}
        self.positions: dict[int, tuple[tuple[int, int], tuple[int, int]]] = {}

    def __getitem__(self, i: int):
        return (
            self.images.get(i, Image.new("RGB", (0, 0))),
            self.positions.get(i, ((0, 0), (0, 0)))
        )

    def __setitem__(
        self,
        key: int,
        value: tuple[Image.Image, tuple[tuple[int, int], tuple[int, int]]]
    ):
        self.images[key] = value[0]
        self.positions[key] = value[1]

    def addToZoom(
        self,
        zoom: int,
        image: Image.Image,
        pos: tuple[tuple[int, int], tuple[int, int]]
    ):
        newPos = [list(i) for i in self.positions.get(zoom, ((0, 0), (0, 0)))]
        if pos[0][0] < self.positions.get(zoom, ((0, 0), (0, 0)))[0][0]:
            newPos[0][0] = pos[0][0]
        if pos[0][1] < self.positions.get(zoom, ((0, 0), (0, 0)))[0][1]:
            newPos[0][1] = pos[0][1]
        if pos[1][0] > self.positions.get(zoom, ((0, 0), (0, 0)))[1][0]:
            newPos[1][0] = pos[1][0]
        if pos[1][1] > self.positions.get(zoom, ((0, 0), (0, 0)))[1][1]:
            newPos[1][1] = pos[1][1]
        newSize = (
            int(ceil(semC(
                abs(newPos[0][0] - newPos[1][0]),
                (0, 0),
                scaleZoomKoeficient**zoom
            ))),
            int(ceil(semYC(
                abs(newPos[0][1] - newPos[1][1]),
                (0, 0),
                scaleZoomKoeficient**zoom
            )))
        )
        newImage = Image.new("RGB", newSize)
        positions = (
            self.positions.get(zoom, ((0, 0), (0, 0)))[0]
            + self.positions.get(zoom, ((0, 0), (0, 0)))[1]
        )
        print(newSize, self.images.get(zoom, Image.new("RGB", (0, 0))))
        print(tuple(map(int, (
                semC(
                    positions[0],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    positions[1],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semC(
                    positions[2],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    positions[3],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                )))
            ))
        newImage.paste(
            self.images.get(zoom, Image.new("RGB", (0, 0))),
            box=tuple(map(int, (
                semC(
                    positions[0],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    positions[1],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semC(
                    positions[2],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    positions[3],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                )))
            )
        )
        newImage.paste(
            image,
            box=tuple(map(int, (
                semC(
                    pos[0][0],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    pos[0][1],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semC(
                    pos[1][0],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                ),
                semYC(
                    pos[1][1],
                    (0, 0),
                    scaleZoomKoeficient**zoom
                )))
            )
        )
        self[zoom] = newImage, newPos

    def draw(self, zoom, surface=display):
        pos = self.positions.get(zoom, ((0, 0), (0, 0)))
        drawImage(
            self.images.get(zoom, Image.new("RGB", (1, 1))),
            (sem(pos[0][0]), semY(pos[0][1]))
        )


def hsvToRgb(h, s, v):
    return tuple(int(i*255) for i in hsv_to_rgb(h/360, s/100, v/100))


def posNegZer(n):
    if n == 0:
        return 0
    elif n < 0:
        return -1
    elif n > 0:
        return 1
    raise ValueError


def isFunction(f):
    def _(): True
    if type(f) in (type(_), type(len)):
        return True
    return False


def roundToNearest(n, base):
    return base*round(n/base)


def pilImageToSurface(pilImage: Image):
    return pg.image.fromstring(
        pilImage.tobytes(), pilImage.size, pilImage.mode).convert()


def tam(x):
    x -= getOfset()[0]
    return (x/(size[0]/RES))*scale


def tamY(y):
    y -= getOfset()[1]
    return (y/(size[1]/RES))*scale


def sem(xr):
    return (xr*(size[0]/RES))/scale + getOfset()[0]


def semY(yr):
    return (yr*(size[1]/RES))/scale + getOfset()[1]


def getOfset():
    return (
        size[0]/2-position[0],
        size[1]/2-position[1]
    )


def visibleCoords():
    return (
        (tam(0), tam(size[0])),
        (tamY(0), tamY(size[1]))
    )


def tamC(x, pos, scale):
    x -= getOfsetC(pos)[0]
    return (x/(size[0]/RES))*scale


def tamYC(y, pos, scale):
    y -= getOfsetC(pos)[1]
    return (y/(size[1]/RES))*scale


def semC(xr, pos, scale):
    return (xr*(size[0]/RES))/scale + getOfsetC(pos)[0]


def semYC(yr, pos, scale):
    return (yr*(size[1]/RES))/scale + getOfsetC(pos)[1]


def getOfsetC(pos):
    return (size[0]/2-pos[0], size[1]/2-pos[1])


def drawFunc(func, color=WHITE, surface=display):
    for x in range(size[0]):
        xr = tam(x)
        # print(func(xr))
        pg.draw.line(
            surface,
            color(func(xr)) if isFunction(color) else color,
            (x, semY(func(xr))),
            (x+1, semY(func(tam(x+1))))
        )


def drawAreaFunc(func, color=WHITE, surface=display):
    for x in range(size[0]):
        for y in range(size[1]):
            c = color(tam(x), tamY(y)) if isFunction(color) else color
            pg.draw.line(surface, c, func(x, y), func(x, y))


def drawAreaFuncImg(func, color=WHITE, surface=display):
    img = Image.new("RGB", size)
    imgDraw = ImageDraw.Draw(img)
    for x in range(size[0]):
        for y in range(size[1]):
            fx, fy = func(x, y)
            imgDraw.point(
                (fx, fy),
                color(tam(x), tamY(y)) if isFunction(color) else color
            )

    surface.blit(pilImageToSurface(img), (0, 0))


def drawImage(image, coords=(0, 0), surface: pg.Surface = display):
    surface.blit(pilImageToSurface(image), coords)


def drawAxis(color=WHITE, surface=display):
    ofset = getOfset()
    pg.draw.line(surface, color, (0, ofset[1]), (size[0], ofset[1]))
    pg.draw.line(surface, color, (ofset[0], 0), (ofset[0], size[1]))
    seeX, seeY = visibleCoords()
    for i in range(floor(seeX[0])*10, ceil(seeX[1])*10):
        pg.draw.line(
            surface,
            color,
            (sem(i/10), ofset[1]-3),
            (sem(i/10), ofset[1]+3)
        )
    for i in range(floor(seeY[0])*10, ceil(seeY[1])*10):
        pg.draw.line(
            surface,
            color,
            (ofset[0]-3, semY(i/10)),
            (ofset[0]+3, semY(i/10))
        )

    for i in range(floor(seeX[0]), ceil(seeX[1])):
        pg.draw.line(
            surface,
            color,
            (sem(i), ofset[1]-10),
            (sem(i), ofset[1]+10)
        )
    for i in range(floor(seeY[0]), ceil(seeY[1])):
        pg.draw.line(
            surface,
            color,
            (ofset[0]-10, semY(i)),
            (ofset[0]+10, semY(i))
        )


@cache
def deCasteljau(t, *points):
    while len(points) > 1:
        # print(len(points))
        newPoints = [points[0]]
        for i, point in enumerate(points):
            if i == len(points)-1:
                continue
            newPoints.append(lerp(point, points[i+1], t))
        # print(len(newPoints), newPoints)
        points = newPoints[1:]
    return points[0]


def drawBezier(surface, color, *points):
    for t in range(10000):
        point1 = deCasteljau(t/10000, *points)
        point2 = deCasteljau((t+1)/10000, *points)
        pg.draw.line(
            surface,
            color,
            (sem(point1[0]), semY(point1[1])),
            (sem(point2[0]), semY(point2[1]))
        )


def getElipsePoints(x, x0, y0, a, b):
    try:
        d = b*math.sqrt(a**2-x0**2+2*x*x0-x**2)/a
    except ValueError:
        return None
    return y0 + d, y0 - d


def drawElipse(x0, y0, a, b, color=WHITE, surface=display):
    for x in range(size[0]):
        xr = tam(x)
        point1 = getElipsePoints(xr, x0, y0, a, b)
        point2 = getElipsePoints(tam(x+1), x0, y0, a, b)
        if point1 is None or point2 is None:
            continue
        for i, p in enumerate(point1):
            pg.draw.line(
                surface,
                color(p) if isFunction(color) else color,
                (x, semY(p)),
                (x+1, semY(point2[i]))
            )


timeDelta = 0
frame = 0


def main():
    global position, scale, RES, zoom, size, display, running, timeDelta
    global frame, lastImageBuffer

    oldState = None

    mouse = Mouse()
    imageBuffer = ImageBuffer()
    while running:
        for u in pg.event.get():
            if u.type == pg.QUIT:
                running = False
            elif u.type == pg.VIDEORESIZE:
                size = (u.w, u.h)
                display = pg.display.set_mode(size)
            elif u.type == pg.MOUSEBUTTONDOWN:
                pg.mouse.get_rel()
                mouse.draging = True
            elif u.type == pg.MOUSEBUTTONUP:
                mouse.draging = False
                mouse.lastDragFrame = frame
            elif u.type == 1027:
                oldPos = (
                    tam(position[0] + getOfset()[0]),
                    tamY(position[1] + getOfset()[1])
                    # tam(pg.mouse.get_pos()[0]),  # + getOfset()[0]),
                    # tamY(pg.mouse.get_pos()[1])  # + getOfset()[1])
                )
                zoom -= posNegZer(u.y)
                if scale < 300:
                    scale = scaleZoomKoeficient**zoom
                else:
                    scale -= u.y/10
                if scale == 0:
                    scale = 0.1
                scale = float("%.3g" % scale)
                position = (
                    sem(oldPos[0]) - getOfset()[0],
                    semY(oldPos[1]) - getOfset()[1]
                )

        if mouse.draging:
            mouseMov = pg.mouse.get_rel()
            position = (
                position[0] - roundToNearest(
                    mouseMov[0],
                    positionSnap
                ),
                position[1] - roundToNearest(
                    mouseMov[1],
                    positionSnap
                )
            )

        startTime = time.time()

        if (
            oldState != (position, RES, scale, zoom, size)
            or mouse.lastDragFrame >= frame-2
        ):
            display.fill(BLACK)

            drawFunc(
                lambda x: 1.1**x,
                lambda x: hsvToRgb(x*10 % 360, 100, 100)
            )

            drawBezier(
                display, LIME,
                (0, 0), (0, 100), (10, 10), (100, 0), (0, 0), (0, 100)
            )

            drawElipse(0, 0, 2, 1)

            drawAxis()

            imageBuffer.draw(zoom, display)

            # print(f"Took: {time.time() - startTime}")

        pg.display.update()
        oldState = (position, RES, scale, zoom, size)
        # print(oldState, visibleCoords(), getOfset())
        # print(display.get_buffer().raw)
        # print(pg.image.frombuffer(display.get_buffer(), size, "RGBA"))
        startTime = time.time()
        # print(getMandelBrotImageMP(size))
        timeDelta = time.time() - startTime
        # print(f"Took {timeDelta}")

        frame += 1


if __name__ == "__main__":
    main()
    pass
