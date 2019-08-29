from unittest import TestCase
from dlcws import util


class TestUtil(TestCase):
    def test_load_data(self):
        self.fail()

    def test_label_segmented_text(self):
        sentence = '叶利钦  重申  不  谋求  连任  '
        chars, labels = util.label_segmented_text(sentence)
        assert chars == ['叶', '利', '钦', '重', '申', '不', '谋', '求', '连', '任']
        assert labels == ['B', 'M', 'E', 'B', 'E', 'S', 'B', 'E', 'B', 'E']
        assert len(chars) == len(labels)
