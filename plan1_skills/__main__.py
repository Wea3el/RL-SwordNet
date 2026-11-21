"""python -m plan1_skills: inspect, acquire, and prepare candidate templates."""
import argparse
import json
from pathlib import Path

from .library import DEFAULT_CACHE, SkillLibrary, build_library, fetch_sources, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List the three candidate skills without downloading")
    fetch = sub.add_parser("fetch", help="Download fixed, checksum-verified sources after reviewing the notice")
    fetch.add_argument("--acknowledge-source-notice", action="store_true")
    fetch.add_argument("--source-dir", type=Path, default=DEFAULT_CACHE / "sources")
    prepare = sub.add_parser("prepare", help="Convert verified local sources into numerical task-space templates")
    prepare.add_argument("--source-dir", type=Path, default=DEFAULT_CACHE / "sources")
    prepare.add_argument("--output", type=Path, default=DEFAULT_CACHE / "templates")
    inspect = sub.add_parser("inspect", help="Sample a normalized phase; does not control a robot")
    inspect.add_argument("--library-dir", type=Path, default=DEFAULT_CACHE / "templates")
    inspect.add_argument("--skill", choices=["guard", "stab", "parry_up"], default="guard")
    inspect.add_argument("--phase", type=float, default=0.)
    args = parser.parse_args()
    try:
        if args.command == "list":
            manifest = load_manifest()
            print(manifest["status"])
            for skill in manifest["skills"]:
                print(f"{skill['action_id']}: {skill['id']} ({skill['kind']})")
        elif args.command == "fetch":
            for path in fetch_sources(args.source_dir, acknowledge_source_notice=args.acknowledge_source_notice):
                print(path)
        elif args.command == "prepare":
            metadata = build_library(args.source_dir, args.output)
            print(f"Prepared {len(metadata['skills'])} candidate templates in {args.output}")
            print(metadata["warning"])
        else:
            library = SkillLibrary(args.library_dir)
            position, direction = library.sample(args.skill, args.phase)
            print(json.dumps({"skill": args.skill, "phase": args.phase,
                              "grip_delta_m_in_skill_frame": position.tolist(),
                              "blade_direction_in_skill_frame": direction.tolist(),
                              "robot_ready": False}, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"plan1_skills: {exc}\n")


if __name__ == "__main__":
    main()
