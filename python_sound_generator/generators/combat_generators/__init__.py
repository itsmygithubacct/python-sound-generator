"""Combat SFX generators, importable as a package.

The CLI does not need this file -- ``combat_sfx.py`` puts its own directory on
``sys.path`` and imports ``lib`` directly, which still works. It exists so other
generators can reach the renderers without that trick, since putting this
directory on ``sys.path`` would publish a module named ``lib`` to the whole
process.

``unarmed_foley_generator`` is the caller: it owns the flesh impacts and borrows
the swings and energy cues from here. The dependency only points that way. This
package stays standard-library-only, and importing it costs the caller nothing.
"""
