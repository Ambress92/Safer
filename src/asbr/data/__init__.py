"""Data pipeline: loading, intersections, and noise injection."""

from asbr.data.loader import FairnessDataset, prepare_split
from asbr.data.adult import AdultDataset
from asbr.data.compas import CompasDataset
from asbr.data.german import GermanDataset
from asbr.data.diabetes import DiabetesDataset
from asbr.data.folktables import ACSIncomeDataset
from asbr.data.celeba import CelebADataset
from asbr.data.intersections import (
    build_subgroup_masks,
    masks_to_torch,
    subgroup_summary,
)
from asbr.data.noise_injection import inject_noise_groups
