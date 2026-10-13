"""dev entrypoint: python -m logitscope [--tiny | --replay FILE] [--port 8000]

--tiny serves the in-process random model — zero downloads, instant startup.
--replay performs a recorded session fixture instead of running any model at
all (no torch import on that path, so it starts in well under a second).
"""

import argparse

import uvicorn


def main() -> None:
    ap = argparse.ArgumentParser("logitscope")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--tiny", action="store_true",
                    help="serve the tiny random model (no downloads)")
    ap.add_argument("--replay", metavar="FILE",
                    help="perform a recorded session fixture (no model at all)")
    args = ap.parse_args()

    # imports stay inside the branches: the replay path must not pay the
    # multi-second torch import just to stream canned bytes
    if args.replay:
        from .replay import create_replay_app

        app = create_replay_app(args.replay)
    else:
        from .app import create_app
        from .service import Service

        svc = Service()
        if args.tiny:
            from .tinymodel import build_tiny_bundle

            svc.register_bundle("tiny-random", build_tiny_bundle())
        app = create_app(svc)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
