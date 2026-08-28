"""Export one subject feature file per Stage 1 LOSO fold."""

from subject_nirs.stage1.exporter import build_parser, main


if __name__ == "__main__":
    main(build_parser().parse_args())
