from utils.file_name import make_filename_safe


class TestMakeFilenameSafe:
    def test_spaces_and_punctuation(self):
        assert make_filename_safe("Going beyond: panaceas!") == "Going_beyond_panaceas"

    def test_ligature_is_expanded_not_dropped(self):
        """PDF text often encodes "fi" as the single ligature character U+FB01."""
        title = "How populations cohere: ﬁve rules for cooperation"
        assert (
            make_filename_safe(title)
            == "How_populations_cohere_five_rules_for_cooperation"
        )

    def test_em_and_en_dashes_become_hyphens(self):
        assert make_filename_safe("Outcomes—Past, Present") == "Outcomes-Past_Present"
        assert make_filename_safe("pages 1–5") == "pages_1-5"

    def test_accents_without_ascii_equivalent_are_removed(self):
        # NFKC keeps precomposed accents, which the ASCII filter then drops
        assert make_filename_safe("Dürer") == "Drer"

    def test_collapses_and_strips_underscores(self):
        assert make_filename_safe("  __a   b__  ") == "a_b"
