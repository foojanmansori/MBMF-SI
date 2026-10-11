import mbmf_q1_infrastructure_v8 as pub


def test_v8_self_checks():
    pub.scientific_core_immutability_test_v8()
    pub.ablation_uniqueness_test_v8()
