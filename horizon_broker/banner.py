"""The broker wordmark.

The README, ``horizon-broker version`` and the interactive banner printed by
``serve`` all draw from one artwork so the three can never drift apart. It is
data on purpose: only data can guarantee they stay identical, and the printing
side decides when a banner is appropriate - a TTY only, so a machine reading
the log stream never has to filter ASCII art out of it.
"""

from __future__ import annotations

__all__ = ["BANNER"]

BANNER = r"""
┌──────────────────────────────────────────────────────────────────────────────┐
│  _   _  ___  ____  ___ ________  _   _   ____  ____   ___  _  _______ ____   │
│ | | | |/ _ \|  _ \|_ _|__  / _ \| \ | | | __ )|  _ \ / _ \| |/ / ____|  _ \  │
│ | |_| | | | | |_) || |  / / | | |  \| | |  _ \| |_) | | | | ' /|  _| | |_) | │
│ |  _  | |_| |  _ < | | / /| |_| | |\  | | |_) |  _ <| |_| | . \| |___|  _ <  │
│ |_| |_|\___/|_| \_\___/____\___/|_| \_| |____/|_| \_\\___/|_|\_\_____|_| \_\ │
│ ················sleep-time compute for a fragile edge fleet················· │
└──────────────────────────────────────────────────────────────────────────────┘
""".lstrip("\n")

# The fleet at sunset: the README "Why" section opens with this so the
# prose about solar trailers and thermal kills has a face to point at.
HORIZON_SCENE = r"""
                                \   |   /
                                 \  |  /
                             ~~~~ ( • ) ~~~~          <- the horizon
                 ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                  ┌───────────┐    ┌───────────┐    ┌───────────┐
                  │   M3 Max  │    │   Pi 5    │    │  Trailer  │
                  │  prefill  │    │  decode   │    │   solar   │
                  └───────────┘    └───────────┘    └───────────┘
                      ~ heartbeat + checkpoint to EdgeRecord ~
"""
