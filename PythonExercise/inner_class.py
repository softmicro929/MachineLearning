#!/usr/bin/env python
class Car:
    class Door:
        def open(self):
            print("open door")

    class Wheel:
        def run(self):
            print("car run")


if __name__ == "__main__":
    car = Car()
    backDoor = Car.Door();
    frontDoor = Car.Door();
    backDoor.open()
    frontDoor.open()

    wheel = Car.Wheel()
    wheel.run()