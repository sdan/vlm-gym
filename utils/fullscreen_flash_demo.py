"""Full-screen flashing TUI demo using curses.

- Fills the terminal, animates a header, progress bar, and a colorized
  activity grid to simulate "processing" with flashing colors.
- No external dependencies; press 'q' to quit.
"""

from __future__ import annotations

import curses
import random
import time
from typing import List, Tuple


SPINNER = "-\\|/"


def _init_colors() -> List[int]:
    curses.start_color()
    try:
        curses.use_default_colors()
    except Exception:
        pass

    # Build a small palette of high-contrast pairs for flashing.
    # We keep it conservative to work on most terminals.
    pairs: List[Tuple[int, int]] = [
        (curses.COLOR_BLACK, curses.COLOR_GREEN),
        (curses.COLOR_BLACK, curses.COLOR_YELLOW),
        (curses.COLOR_WHITE, curses.COLOR_BLUE),
        (curses.COLOR_BLACK, curses.COLOR_CYAN),
        (curses.COLOR_BLACK, curses.COLOR_MAGENTA),
        (curses.COLOR_BLACK, curses.COLOR_RED),
        (curses.COLOR_WHITE, curses.COLOR_BLACK),
    ]
    ids: List[int] = []
    pair_id = 1
    for fg, bg in pairs:
        try:
            curses.init_pair(pair_id, fg, bg)
            ids.append(pair_id)
            pair_id += 1
        except Exception:
            # Some terminals may not support many pairs; stop on failure
            break
    if not ids:
        # Ensure at least one usable pair exists
        curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
        ids = [1]
    return ids


def _draw_header(stdscr, cols: int, tick: int, color_ids: List[int]) -> None:
    spin = SPINNER[tick % len(SPINNER)]
    msg = f" Processing {spin}  Press 'q' to quit "
    pad = max(0, cols - len(msg) - 2)
    color = curses.color_pair(color_ids[tick % len(color_ids)]) | curses.A_BOLD
    stdscr.addstr(0, 0, " " * cols, color)
    stdscr.addstr(0, 1, msg, color)
    if pad:
        stdscr.addstr(0, 1 + len(msg), " " * pad, color)


def _draw_progress(stdscr, row: int, cols: int, tick: int, color_ids: List[int]) -> None:
    width = max(10, cols - 12)
    pos = tick % width
    color = curses.color_pair(color_ids[(tick // 2) % len(color_ids)]) | curses.A_REVERSE
    bar_chars = [" "] * width
    for i in range(width):
        if i <= pos:
            bar_chars[i] = "█"
        elif (i // 4 + tick) % 2 == 0:
            bar_chars[i] = "▒"
        else:
            bar_chars[i] = "░"
    stdscr.addstr(row, 1, "Progress:", curses.A_DIM)
    stdscr.addstr(row, 11, "".join(bar_chars), color)


def _draw_grid(stdscr, start_row: int, rows: int, cols: int, tick: int, color_ids: List[int]) -> None:
    # Use a coarse grid of blocks; animate color and sparsity.
    # Each cell is one character wide for maximal coverage.
    rng = random.Random(1337 + tick)
    cycle = max(1, len(color_ids))
    for r in range(rows):
        base = (r * 7 + tick) % cycle
        for c in range(cols):
            # Flash intensity pattern with spatial-temporal variation
            phase = (base + c * 11) % cycle
            color = curses.color_pair(color_ids[phase])
            if ((r + c + tick) % 9) == 0:
                color |= curses.A_BOLD
            ch = "█" if rng.random() < 0.75 else ("▓" if rng.random() < 0.6 else "░")
            try:
                stdscr.addstr(start_row + r, c, ch, color)
            except curses.error:
                # Ignore drawing past bounds (e.g., small terminals or last column)
                pass


def _draw_footer(stdscr, last_row: int, cols: int) -> None:
    msg = " Rendering full-screen demo | q=quit | +/-=speed | r=rainbow "
    pad = max(0, cols - len(msg) - 2)
    stdscr.addstr(last_row, 0, " " * cols, curses.A_REVERSE)
    stdscr.addstr(last_row, 1, msg, curses.A_REVERSE)
    if pad:
        stdscr.addstr(last_row, 1 + len(msg), " " * pad, curses.A_REVERSE)


def _loop(stdscr) -> None:
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(0)
    color_ids = _init_colors()

    tick = 0
    delay = 0.05  # ~20 FPS
    rainbow = False

    while True:
        stdscr.erase()
        rows, cols = stdscr.getmaxyx()
        if rows < 6 or cols < 20:
            stdscr.addstr(0, 0, "Terminal too small; resize please.")
            stdscr.refresh()
            time.sleep(0.2)
            ch = stdscr.getch()
            if ch in (ord("q"), ord("Q")):
                break
            continue

        # Optionally rotate color palette to create a rainbow effect
        palette = color_ids
        if rainbow and len(color_ids) > 1:
            shift = (tick // 2) % len(color_ids)
            palette = color_ids[shift:] + color_ids[:shift]

        # Header / progress
        _draw_header(stdscr, cols, tick, palette)
        _draw_progress(stdscr, 2, cols, tick, palette)

        # Activity grid fills remaining space between header and footer
        grid_top = 4
        grid_rows = max(1, rows - grid_top - 2)
        grid_cols = cols
        _draw_grid(stdscr, grid_top, grid_rows, grid_cols, tick, palette)

        # Footer
        _draw_footer(stdscr, rows - 1, cols)

        stdscr.refresh()

        # Input handling
        ch = stdscr.getch()
        if ch == ord("q") or ch == ord("Q"):
            break
        elif ch in (ord("+"), ord("=")):
            delay = max(0.005, delay * 0.8)
        elif ch in (ord("-"), ord("_")):
            delay = min(0.25, delay * 1.25)
        elif ch in (ord("r"), ord("R")):
            rainbow = not rainbow

        tick += 1
        time.sleep(delay)


def main() -> None:
    curses.wrapper(_loop)


if __name__ == "__main__":
    main()

