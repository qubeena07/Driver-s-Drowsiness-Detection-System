import threading
def inc():
    a = 5
    b=4
    import time

    t_end = time.time() + 15
    while time.time() < t_end:
        import drowsiness_test
    return a+b

def test_answer():
    assert inc()   


