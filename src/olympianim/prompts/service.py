"""Prompt management service backed by SQLite."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from olympianim.config import DATABASE_PATH
from olympianim.database.models import (
    ProjectPromptRecord,
    PromptRecord,
    PromptVersionRecord,
)
from olympianim.database.repository import ProjectRepository
from olympianim.prompts.defaults import (
    DEFAULT_PROMPTS,
    DEFAULT_PROMPTS_BY_IDENTITY,
)
from olympianim.prompts.validator import (
    PromptValidationResult,
    render_prompt_template,
    validate_prompt_template,
)
from olympianim.prompts.variables import AGENT_SPECS, AgentSpec, variables_for_agent

_LEGACY_DEFAULT_HASHES = frozenset(
    {
        "165d7bd5331997b04d73ba7c394553525f3a66917eecf8f5bfe5d139fe8f8032",
        "aada5f7b97154d871adf1562bd7e0bb66d2b2afc408aeb2477eaeee77eac224a",
        "0a05e33d899ad3cac6210cf377960a4aa78dc967801e4ae937d6e3c49e35691e",
        "a14a4ab28d424debc60399c1e5bae56c41f0169467aa3908afee2638f2145eae",
        "fc41999f6e2b3ea2b9135e1ebc324adfb361291cddadfe690deee47017f79fb5",
        "7e054b8b9dd40992cab1f34d4383f9dce0418a6c803f21dc0e2b4a3cf2584f94",
        "cdbcbf3e47b31e9b17762d31ceb31b1f10488e752ef57e44a7d732cacee40900",
        "d73207446214ed6669fac1025d7d976352e2d879935ea2bda846c0bf2fd82793",
        "f5abe3ae7219a57e4e4f24d86d268dd10884d80cb87abf348b6e0a1f504d35ec",
        # Builder defaults immediately before the generic mixed-math contract.
        "4a428d282567474124c95010b87a4e4d51b99ec79ba2167f00cc41ed47416913",
        "fd0c0dafd7524b5b82921428b608e72869221eeac2b540ce9d43dfaf5be399a6",
        # Builder defaults immediately before atomic narration choreography.
        "4674e599ad1ffc54a24bc3bd97840ae756b7e14fd3f1bbc2f56112753c201f19",
        "b0899b4b296923407c3fef72616d81b894813db995d2827c4c71c1828ba10d57",
        # Defaults immediately before context-aware visual choreography.
        "2589f8983d5e1267376aa73843e48fdecf19bc698574469b12129101afdb1aff",
        "858d85a7ab29eafb795480b760f6b1f3aace2e7afc2695cc2bcad498d91e6572",
        "57fa3ebe115489ef47ae6a735ea9b53c1d3ddb9e2a412b1626d947d5d306c760",
        "420ab85b19618c4f3397a1263cd396a39e09ee8471421f6e97f1a015fe70525f",
        "8b9af9c4ce32581077c06dd739c28bb9d1accfad05bddfae83ecd962a9ce5a53",
        # Presentation planner immediately before private solution guidance.
        "85b000b403466cd88c6872a46c11ca1bf25ff2ed1b6663530434857cfe700592",
        # Solver immediately before solution images became method-binding.
        "bf07fa7e637ff3cc4935d3a70be4af67d4da55fd52bf70610f14be5dbebd9e0d",
        # Solver before image-only instructions moved to runtime context.
        "a3569fa8fd1c2294a7ecb53351fa0ba51df7d5f0c25efe99bd74e79b59ee20c4",
        # Mathematical-decision prompts immediately before the rigor contract.
        "d1987af4a2c504ac16043269caea313ee9f0b75f3127d0c736728892a0e49d5e",
        "4359db9daf4713acd49ca1582b70adf205ae508e231471da4967edff5e86e054",
        "5d1a2c02c90025fb22e307217341d586af7f98431942962c5282635da80694e5",
        # Code-producing defaults before the strict Python output contract.
        "36b7fba9d22a3f7e17a4769ec50f254cd11edc3a8eb19b7da0da78581667e6eb",
        "f38fbed8e8b25869d000cfef4543e9f5f792bc9ee9ba9bc4cc1c433d8da7b84a",
        "8e5f5f1cabd030a3592d5e96c0769256ca9419f8d0a7ef79a9e5c34a6a9ebae8",
        "ee6c6bfb6cf96c1359c8a5f575ad3e7694f2948f8416d2513e0481c229ea766c",
        # Short-lived revisions that also constrained Python symbol spelling.
        "901ba6634d603bf96e7da824d06e524e2188d5afc48884edbdb16e2b5e42b98b",
        "6dd7d614e32c5852912776eb88b1efcd2870ca4b36d499cc77a14ba536c50214",
        "259a175cf8389d02a3043e873c11796b128cbb0481323032821abd1d5092b6bc",
        "17bf0bfbbfb95b914ea5a3d864668527ead737359b8336167e03d1664b647045",
        # Code-producing defaults before their output became schema-enforced.
        "71b3879e214a0ceccfbd994a20d8da9e977e366de8b82cbcacd258208fb1b307",
        "2f8feac05ae5563d593e6c6ed7f59b31dc9065edc7e20fc72e7dfe30665f7f3d",
        "126a5c2f548462af2596749056cc703d9fb2ba4012fc4c8b03f00047280967bb",
    }
)


@dataclass(frozen=True)
class PromptWithVersion:
    """Prompt metadata together with its latest template version."""

    prompt: PromptRecord
    latest_version: PromptVersionRecord


class PromptService:
    """Use case layer for prompt templates, versions and snapshots."""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self.repository = repository or ProjectRepository(DATABASE_PATH)

    def ensure_default_prompts(self) -> None:
        """Seed and safely upgrade built-in prompts without overwriting user edits."""
        existing = {
            (prompt.agent_type, prompt.name): prompt for prompt in self.repository.list_prompts()
        }
        for default in DEFAULT_PROMPTS:
            key = (default.agent_type, default.name)
            if key in existing:
                self._upgrade_unchanged_default(existing[key], default.template_text)
                continue
            prompt = self.repository.create_prompt(
                name=default.name,
                agent_type=default.agent_type,
                description=default.description,
                is_default=True,
            )
            self.repository.add_prompt_version(prompt.id, template_text=default.template_text)
            self.repository.set_setting(
                self._default_hash_key(prompt.id),
                self._sha256(default.template_text),
            )

    def _upgrade_unchanged_default(self, prompt: PromptRecord, template_text: str) -> None:
        """Upgrade any known built-in revision while preserving every user edit."""
        if not prompt.is_default:
            return
        latest = self.repository.get_latest_prompt_version(prompt.id)
        if latest is None:
            return
        current_hash = self._sha256(template_text)
        latest_hash = self._sha256(latest.template_text)
        state_key = self._default_hash_key(prompt.id)
        previously_applied = self.repository.get_setting(state_key)
        known_builtins = _LEGACY_DEFAULT_HASHES | (
            {previously_applied} if previously_applied else set()
        )
        if latest_hash != current_hash and latest_hash in known_builtins:
            self.repository.add_prompt_version(prompt.id, template_text=template_text)
        self.repository.set_setting(state_key, current_hash)

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _default_hash_key(prompt_id: str) -> str:
        return f"prompt.default_sha256.{prompt_id}"

    def list_agents(self) -> tuple[AgentSpec, ...]:
        """Return all prompt-owning agents."""
        return AGENT_SPECS

    def list_prompts(self, agent_type: str | None = None) -> list[PromptWithVersion]:
        """List prompts with their latest version, optionally filtered by agent."""
        self.ensure_default_prompts()
        prompts = self.repository.list_prompts()
        if agent_type is not None:
            prompts = [prompt for prompt in prompts if prompt.agent_type == agent_type]
        return [
            PromptWithVersion(prompt=prompt, latest_version=version)
            for prompt in prompts
            if (version := self.repository.get_latest_prompt_version(prompt.id)) is not None
        ]

    def get_prompt(self, prompt_id: str) -> PromptWithVersion | None:
        """Return one prompt with its latest template."""
        prompt = self.repository.get_prompt(prompt_id)
        if prompt is None:
            return None
        version = self.repository.get_latest_prompt_version(prompt.id)
        if version is None:
            return None
        return PromptWithVersion(prompt=prompt, latest_version=version)

    def save_prompt_version(
        self,
        prompt_id: str,
        template_text: str,
    ) -> tuple[PromptVersionRecord | None, PromptValidationResult]:
        """Validate and save a new version for an existing prompt."""
        prompt = self.repository.get_prompt(prompt_id)
        if prompt is None:
            raise ValueError(f"Prompt not found: {prompt_id!r}")
        validation = validate_prompt_template(prompt.agent_type, template_text)
        if not validation.valid:
            return None, validation
        version = self.repository.add_prompt_version(prompt_id, template_text=template_text)
        return version, validation

    def duplicate_prompt(self, prompt_id: str, new_name: str) -> PromptWithVersion:
        """Duplicate a prompt and copy its latest version."""
        source = self.get_prompt(prompt_id)
        if source is None:
            raise ValueError(f"Prompt not found: {prompt_id!r}")
        prompt = self.repository.create_prompt(
            name=new_name,
            agent_type=source.prompt.agent_type,
            description=f"Copia de {source.prompt.name}",
            is_default=False,
        )
        version = self.repository.add_prompt_version(
            prompt.id,
            template_text=source.latest_version.template_text,
        )
        return PromptWithVersion(prompt=prompt, latest_version=version)

    def restore_default_prompt(self, prompt_id: str) -> PromptVersionRecord:
        """Append the built-in default template as the prompt's newest version."""
        prompt = self.repository.get_prompt(prompt_id)
        if prompt is None:
            raise ValueError(f"Prompt not found: {prompt_id!r}")
        default = DEFAULT_PROMPTS_BY_IDENTITY.get((prompt.agent_type, prompt.name))
        if default is None:
            candidates = [
                item
                for item in DEFAULT_PROMPTS
                if item.agent_type == prompt.agent_type and prompt.name.startswith(item.name)
            ]
            if len(candidates) == 1:
                default = candidates[0]
            else:
                agent_defaults = [
                    item for item in DEFAULT_PROMPTS if item.agent_type == prompt.agent_type
                ]
                if len(agent_defaults) == 1:
                    default = agent_defaults[0]
        if default is None:
            raise ValueError("Não foi possível identificar o prompt padrão de origem desta cópia.")
        return self.repository.add_prompt_version(prompt.id, template_text=default.template_text)

    def variables_for_agent(self, agent_type: str) -> tuple[str, ...]:
        """Return variables available to one agent."""
        return variables_for_agent(agent_type)

    def validate(self, agent_type: str, template_text: str) -> PromptValidationResult:
        """Validate a template without saving it."""
        return validate_prompt_template(agent_type, template_text)

    def save_project_prompt_snapshot(
        self,
        project_id: str,
        *,
        agent_type: str,
        prompt_id: str,
        values: dict[str, object],
    ) -> ProjectPromptRecord:
        """Render and persist the prompt version used by a project."""
        prompt = self.get_prompt(prompt_id)
        if prompt is None:
            raise ValueError(f"Prompt not found: {prompt_id!r}")
        rendered = render_prompt_template(prompt.latest_version.template_text, values)
        return self.repository.record_project_prompt(
            project_id,
            agent_type=agent_type,
            prompt_id=prompt_id,
            prompt_version=prompt.latest_version.version,
            rendered_prompt_snapshot=rendered,
        )


def prompt_service_for_database(database_path: Path) -> PromptService:
    """Create a prompt service bound to ``database_path`` for tests."""
    return PromptService(repository=ProjectRepository(database_path))
