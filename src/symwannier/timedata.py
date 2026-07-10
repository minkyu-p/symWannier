#!/usr/bin/env python

import time
import datetime

class TimeData:
    def __init__(self):
        self.start_time = time.time()
        self.start_datetime = datetime.datetime.now()
        self.clock_count = {}
        self.clock_start = {}
        self.clock_total = {}

    @staticmethod
    def format_duration(seconds):
        minutes = int(seconds // 60)
        seconds_rem = seconds - 60 * minutes
        if minutes > 0:
            return "{:d}m {:05.2f}s".format(minutes, seconds_rem)
        return "{:.2f}s".format(seconds_rem)

    def start_clock(self, clock_name):
        self.clock_count[clock_name] = self.clock_count.get(clock_name, 0) + 1
        self.clock_start[clock_name] = time.time()

    def stop_clock(self, clock_name):
        self.clock_total[clock_name] = time.time() - self.clock_start.get(clock_name, self.start_time) + self.clock_total.get(clock_name, 0)

    def show_clock(self, clock_name):
        total = self.clock_total[clock_name]
        count = self.clock_count[clock_name]
        print(" {:25s}: {:>12s}  ({:>12s} for each call) ".format(
            clock_name,
            self.format_duration(total),
            self.format_duration(total / count),
        ))

    def show_time(self):
        tot_time = time.time() - self.start_time
        print(" total time               : {:>12s}".format(self.format_duration(tot_time)))

    def get_time(self):
        return time.time() - self.start_time

    def show_all(self):
        tot_time = time.time() - self.start_time
        print("")
        print(" total time               : {:>12s}".format(self.format_duration(tot_time)))
        for key in self.clock_count.keys():
            key_tot = self.clock_total.get(key, 0)
            key_count = self.clock_count.get(key, 1)
            print(" {:25s}: {:>12s}  ({:>12s} for each call) ".format(
                key,
                self.format_duration(key_tot),
                self.format_duration(key_tot / key_count),
            ))
