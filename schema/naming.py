"""The naming convention that ties an EOS lab to its generated Matterix scene: one scene
per lab, gym id derived from the lab name, workflow key = EOS task name. from_eos.py names
the scene with this; matterix_backend.py resolves a device's (lab, task) with it -- one
function on both sides, so they can't drift.

Imports nothing on purpose: matterix_backend.py calls this inside the MatterixBackend
actor, whose conda env may not have `eos` (see runtime.py's
`_discover_user_package_dirs()`).
"""


def _pascal(name: str) -> str:
    return "".join(word.capitalize() for word in name.replace("-", "_").split("_"))


def lab_gym_id(lab_name: str) -> str:
    """The gym id of the scene `python -m schema.from_eos` generates for `lab_name`."""
    return f"Matterix-Lab-{_pascal(lab_name)}-v1"
