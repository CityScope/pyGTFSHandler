"""`same_operators`: a feed whose operators all run in other feeds is another export of the same service."""

from pyGTFSHandler.utils.feed_dedupe import feed_agencies, same_operators


def _feed(d, rows):
    d.mkdir()
    (d / "agency.txt").write_text("agency_id,agency_name,agency_url\n" + "\n".join(rows) + "\n")
    return d


def test_operators(tmp_path):
    official = _feed(tmp_path / "official", ["1,Brockton Area Transit Authority (BAT),http://www.ridebat.com/"])
    copy = _feed(tmp_path / "copy", ["444,Brockton Area Transit Authority,https://ridebat.com"])
    other = _feed(tmp_path / "other", ["9,Harvard Shuttles,https://transportation.harvard.edu", "1,BAT,http://ridebat.com"])
    no_url = _feed(tmp_path / "no_url", ["1,Brockton Area Transit Authority (BAT),"])
    assert feed_agencies(official) == {"ridebat.com"}
    assert same_operators(copy, [official])
    assert not same_operators(other, [official])          # also runs an operator the official feeds lack
    assert feed_agencies(no_url) == {"brockton area transit authority"}
    assert not same_operators(tmp_path / "missing", [official])
