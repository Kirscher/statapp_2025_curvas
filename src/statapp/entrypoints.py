def echo(ping):
    print(ping)

def setup():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', nargs='+', required=True, type=int,
                        help="[REQUIRED] Something to say")
    args, unrecognized_args = parser.parse_known_args()
    echo(args.d)