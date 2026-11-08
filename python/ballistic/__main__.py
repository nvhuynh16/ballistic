"""``python -m ballistic doctor``: check this installation (exit 1 on a failed check)."""

import sys


def main(argv=None):
    """Run the command line; ``argv`` defaults to sys.argv[1:].

    Returns the exit status: 0 all checks passed, 1 a check failed, 2 bad usage.
    """
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] != ["doctor"] or len(argv) > 2 or argv[1:] not in ([], ["--no-self-test"]):
        print("usage: python -m ballistic doctor [--no-self-test]", file=sys.stderr)
        return 2
    from .doctor import doctor
    return 0 if doctor(self_test="--no-self-test" not in argv).ok else 1


if __name__ == "__main__":
    sys.exit(main())
