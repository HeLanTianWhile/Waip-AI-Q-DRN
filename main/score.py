import math


def score(mouse_positions, click_positions):
    error = 0
    D = math.hypot(390, 844)
    KEY_W = (43.5, 676.0, 77.0, 712.0)
    KEY_W_CENTER = (60.25, 694.0)
    if click_positions:
        click_errors = []
        w_hit = False
        for x,y in click_positions:
            if KEY_W[0] <= x < KEY_W[2] and KEY_W[1] <= y < KEY_W[3]:
                click_errors.append(0)
                w_hit = True
            else:
                click_errors.append((math.hypot(x - KEY_W_CENTER[0], y - KEY_W_CENTER[1])) / D)
        if w_hit:
            error = 0
        else:
            error = sum(click_errors) / len(click_errors)
    else:
        mouse_position_errors = []
        for x,y in mouse_positions:
            mouse_position_errors.append(math.hypot(x - KEY_W_CENTER[0], y - KEY_W_CENTER[1]) / D)
        error = sum(mouse_position_errors) / len(mouse_position_errors) + 0.5
        
    hit_rate = 1 - error
    return hit_rate
