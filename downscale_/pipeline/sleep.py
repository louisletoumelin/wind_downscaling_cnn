from time import sleep


def sleep_5_mintues():
    return sleep(5*60)


hours_to_sleep = 6

for hours in range(hours_to_sleep):
    for i in range(12):
        print(f"hours: {hours}, i={i}/12")
        sleep_5_mintues()
