from time import sleep


def sleep_5_minutes():
    return sleep(5*60)


hours_to_sleep = 3

for hours in range(hours_to_sleep):
    for i in range(12):
        print(f"hours: {hours}, i={i}/12")
        sleep_5_minutes()
