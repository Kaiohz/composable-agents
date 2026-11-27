from abc import ABC, abstractmethod
from typing import Any

from src.domain.entities.mcp_server_config import McpServerConfig


class McpToolLoader(ABC):
    """Port pour charger des outils depuis des serveurs MCP."""

    @abstractmethod
    async def load_tools(self, configs: list[McpServerConfig], agent_scope: str | None = None) -> list[Any]:
        """Charge les outils depuis les serveurs MCP configures.

        Args:
            configs: Liste des configurations de serveurs MCP.
            agent_scope: Nom de l'agent courant dont l'espace (scope) les
                connexions sortantes HTTP relevent. La plateforme (chargement
                d'outils, jamais un choix du LLM) injecte automatiquement
                l'en-tete ``X-Composable-Agent: {agent_scope}`` sur chaque
                connexion HTTP (REQ-016) et resout le placeholder
                ``${AGENT_NAME}`` des en-tetes configures (REQ-017).
                ``None`` = contexte hors agent (assistant integre, scope
                ``personal``): aucun en-tete injecte.

        Returns:
            Liste des outils charges depuis les serveurs MCP.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Ferme toutes les connexions MCP ouvertes."""
        ...
