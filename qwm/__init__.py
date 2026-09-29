"""Queryable World Models (QWM).

Paper title: "Queryable World Models: Predicting Outcomes, Not Pixels"

North star: given (obs_t, action_t), predict the next QUERYABLE STATE and use
it to plan, beating an identical backbone without state heads on the same
data.

The core query representation (exact frozen layout in schema.md):

    query_state = [grip_xyz, obj_xyz, upright, contact, on_table, ...]

Dreamer/JEPA/Genie dream. QWM answers.
"""

__version__ = "0.1.0"
