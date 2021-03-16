from time import time


class Timer:
    def __init__(self):
        self.entries = {}
        self.start_times = {}

    def start(self, description):
        return self.Instance(self, description)

    def get_average_time(self, description):
        times = self.entries[description]
        return sum(times) / len(times)

    def get_total_time(self, description):
        return sum(self.entries[description])

    class Instance:
        def __init__(self, timer, description):
            self.timer = timer
            self.description = description

        def __enter__(self):
            if self.description in self.timer.start_times:
                raise Exception(
                    "Cannot start timing {}".format(self.description)
                    + " again before finishing the last invocation"
                )
            self.timer.start_times[self.description] = time()

        def __exit__(self, type, value, traceback):
            end = time()
            start = self.timer.start_times[self.description]

            if self.description not in self.timer.entries:
                self.timer.entries[self.description] = []

            self.timer.entries[self.description].append(end - start)
            del self.timer.start_times[self.description]


def main():
    def fact(n):
        if n == 0:
            return 1
        return n * fact(n - 1)

    my_timer = Timer()
    for i in range(10):
        with my_timer.start("fact"):
            print(fact(i))

    print(my_timer.get_average_time("fact"))
    print(my_timer.get_total_time("fact"))
    print(my_timer.entries)

    def bad_fact(n):
        if n == 0:
            return 1
        with my_timer.start("bad_fact"):
            return n * fact(n - 1)

    with my_timer.start("bad_fact"):
        print(bad_fact(10))


if __name__ == "__main__":
    main()
