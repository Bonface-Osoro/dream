from dream.dhSurvey import recode_type, survey_round_label


def test_individual_recode_file_name_gives_ir():
    assert recode_type('UGIR72FL') == 'IR'


def test_recode_type_ignores_case():
    assert recode_type('ugpr72fl') == 'PR'


def test_untargeted_recode_file_name_gives_none():
    assert recode_type('UGMR72FL') is None


def test_round_label_is_second_segment_of_folder_name():
    name = 'UG_2014-15_MIS_09212026_1323_150620'
    assert survey_round_label(name) == '2014-15'


def test_round_label_falls_back_to_full_name_without_underscore():
    assert survey_round_label('uganda') == 'uganda'
