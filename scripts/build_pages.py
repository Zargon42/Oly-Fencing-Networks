"""Build the static app with isolated, public-safe marimo configuration."""
import shutil
import os
import stat
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
output = root/'_site'
# Delete only this repository's generated export directory, never source inputs.
if output.is_symlink() or output.resolve().parent != root or output.name != '_site':
    raise RuntimeError('Unexpected build output path')
def remove_readonly(function, path, error):
    # Exported assets can inherit read-only attributes on Windows/OneDrive.
    if not isinstance(error, PermissionError):
        raise error
    os.chmod(path, stat.S_IREAD | stat.S_IWRITE | stat.S_IEXEC)
    function(path)

if output.exists():
    shutil.rmtree(output, onexc=remove_readonly)
subprocess.run([
    sys.executable, '-m', 'marimo', 'export', 'html-wasm', 'fencing_app.py',
    '-o', str(output), '--mode', 'run', '--no-sandbox', '--force',
], cwd=root/'pages', stdin=subprocess.DEVNULL, check=True, timeout=180)
html = (output/'index.html').read_text(encoding='utf-8')
# marimo 0.23.10 serializes default runtime settings in its WASM export, even
# when local configuration requests auto-run. Set the export's runtime override
# explicitly so a run-mode app does not open as an empty page. Fail on schema
# changes rather than silently shipping a broken site.
marker = '"configOverrides": {}'
assert html.count(marker) == 1, 'Unexpected marimo export configuration schema'
html = html.replace(marker, '"configOverrides": {"runtime": {"auto_instantiate": true}}')
# The worker also consumes userConfig directly on a fresh origin. Keep both
# representations consistent; relying on only configOverrides can leave the
# first published visit blank even though a warm localhost origin works.
assert html.count('"auto_instantiate": false') == 1, 'Unexpected serialized runtime defaults'
html = html.replace('"auto_instantiate": false', '"auto_instantiate": true', 1)
(output/'index.html').write_text(html, encoding='utf-8')
assert '"auto_instantiate": true' in html, 'The exported app must start automatically'
assert '"auto_instantiate": false' not in html
assert (output/'public/analysis.json').read_bytes() == (root/'pages/public/analysis.json').read_bytes()
assert (output/'assets').is_dir()
assert {p.name for p in (output/'public').iterdir()} == {'analysis.json'}, 'Unexpected public data bundled'
(output/'.nojekyll').touch()
# Some marimo distributions include this vendor development note among frontend assets.
(output/'CLAUDE.md').unlink(missing_ok=True)
print('Pages build verified: automatic execution, relative assets, and the exact actual-results snapshot.')
