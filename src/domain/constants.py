"""Shared domain constants.

Single owners for values that several layers need (domain services,
application validation and infrastructure adapters), so no layer has to
import another layer to share a number.
"""

#: Maximum agent-YAML size in BYTES. Write paths (REST and MCP agent
#: creation, via ``application/services/route_validation.read_yaml_bytes``)
#: and the read side (the agent-topology resolver's stored-YAML cap, review
#: NV-1) enforce the SAME bound.
MAX_YAML_SIZE = 1024 * 1024
