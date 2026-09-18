import xml.etree.ElementTree as ET

import pytest


@pytest.fixture
def skin_articulation():
    urdf = ET.Element("robot", name="articulation")
    for name in ("base", "tip"):
        link = ET.SubElement(urdf, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz="0.02 0.01 -0.01")
        ET.SubElement(inertial, "mass", value="2")
        ET.SubElement(inertial, "inertia", ixx="0.11", iyy="0.09", izz="0.13", ixy="0.02", ixz="0.01", iyz="0.015")
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="0.08 0.08 0.08")
    joint = ET.SubElement(urdf, "joint", name="hinge", type="revolute")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="tip")
    ET.SubElement(joint, "origin", xyz="0.2 0 0")
    ET.SubElement(joint, "axis", xyz="0 0 1")
    ET.SubElement(joint, "limit", lower="-1", upper="1", effort="100", velocity="10")
    return ET.tostring(urdf, encoding="unicode")


@pytest.fixture
def skin_collision_union():
    urdf = ET.Element("robot", name="collision_union")
    for name, centers in (("base", ("-0.02 0 0", "0.02 0 0")), ("tip", ("0 0 0",))):
        link = ET.SubElement(urdf, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(inertial, "inertia", ixx="0.01", iyy="0.01", izz="0.01", ixy="0", ixz="0", iyz="0")
        for center in centers:
            collision = ET.SubElement(link, "collision")
            ET.SubElement(collision, "origin", xyz=center)
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", size="0.08 0.08 0.08")
    joint = ET.SubElement(urdf, "joint", name="hinge", type="revolute")
    ET.SubElement(joint, "parent", link="base")
    ET.SubElement(joint, "child", link="tip")
    ET.SubElement(joint, "origin", xyz="0.2 0 0", rpy="0 0 1.5707963267948966")
    ET.SubElement(joint, "axis", xyz="0 0 1")
    ET.SubElement(joint, "limit", lower="-1", upper="1", effort="100", velocity="10")
    return ET.tostring(urdf, encoding="unicode")


@pytest.fixture
def skin_chain():
    urdf = ET.Element("robot", name="skin_chain")
    for name in ("base", "middle", "tip"):
        link = ET.SubElement(urdf, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "mass", value="1")
        ET.SubElement(inertial, "inertia", ixx="0.01", iyy="0.01", izz="0.01", ixy="0", ixz="0", iyz="0")
        collision = ET.SubElement(link, "collision")
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "box", size="0.08 0.08 0.08")
    for parent, child in (("base", "middle"), ("middle", "tip")):
        joint = ET.SubElement(urdf, "joint", name=child, type="revolute")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=child)
        ET.SubElement(joint, "origin", xyz="0.2 0 0")
        ET.SubElement(joint, "axis", xyz="0 0 1")
        ET.SubElement(joint, "limit", lower="-1", upper="1", effort="100", velocity="10")
    return ET.tostring(urdf, encoding="unicode")
