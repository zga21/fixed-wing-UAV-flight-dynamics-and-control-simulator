"""Regenerate committed validation evidence."""

from record_baseline import write_record
from validation_report import main as phase3_validation_main


def main() -> None:
    phase3_validation_main()
    path = write_record()
    print(path)


if __name__ == "__main__":
    main()
