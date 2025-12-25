import pytest
from src.fewshot_core.classes import CLASSES, CLASS_TO_ID, ID_TO_CLASS

def test_class_list_is_fixed():
    """
    Tests that the class list has the correct number of classes.
    """
    assert len(CLASSES) == 13, "There should be exactly 13 fixed classes."

def test_class_to_id_mapping():
    """
    Tests that the CLASS_TO_ID mapping is correct and consistent.
    """
    assert len(CLASS_TO_ID) == 13
    assert CLASS_TO_ID["Airplane"] == 0
    assert CLASS_TO_ID["Watercraft"] == 12
    # Check a class in the middle
    assert CLASS_TO_ID["Lamp"] == 6

def test_id_to_class_mapping():
    """
    Tests that the ID_TO_CLASS mapping is the correct inverse of CLASS_TO_ID.
    """
    assert len(ID_TO_CLASS) == 13
    assert ID_TO_CLASS[0] == "Airplane"
    assert ID_TO_CLASS[12] == "Watercraft"
    assert ID_TO_CLASS[6] == "Lamp"

def test_mappings_are_consistent():
    """
    Ensures that converting from class to ID and back yields the same class.
    """
    for class_name in CLASSES:
        class_id = CLASS_TO_ID[class_name]
        assert ID_TO_CLASS[class_id] == class_name
