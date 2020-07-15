from fragalysis_api.xcextracter.getdata import GetPdbData
from fragalysis_api.xcextracter.xcextracter import xcextracter


def main():
    mpro_summary = xcextracter('Mpro')
    print(mpro_summary)


if __name__ == "__main__":
    main()
