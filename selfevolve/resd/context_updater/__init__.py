# from .ace import ACEContextUpdater
from .playbook_context_updater import PlaybookContextUpdater

# Upstream `ray_trainer.py` still does:
#   from ...context_updater import ACEContextUpdater, PlaybookContextUpdater
# but the original `ace.py` module has been removed from this repo and
# the symbol is never actually used at runtime. Re-export PlaybookContextUpdater
# under the legacy name so the import succeeds.
ACEContextUpdater = PlaybookContextUpdater