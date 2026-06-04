"""Rebuild the face database for the InsightFace (ArcFace) backend.

The legacy dlib database (``faces.pkl``, 128-d Euclidean) is incompatible with
the InsightFace backend, which uses 512-d L2-normalized ArcFace embeddings under
cosine distance. Rather than re-meeting everyone, this tool re-embeds the per-
person face crops that were saved alongside the old db
(``known_faces/pXXX/*.jpg``) and writes a fresh v3 cosine db
(``known_faces/faces_arcface.pkl``).

Usage:
    uv run python migrate_db.py [--db-dir known_faces] [--det-size 640]
                                [--det-thresh 0.5] [--dry-run]

The original ``faces.pkl`` is left untouched, so the dlib backend keeps working.
"""

import argparse
import logging
import os
import re

import cv2
import numpy as np

from face_tracker import FaceDatabase, InsightFaceBackend

logger = logging.getLogger("migrate_db")

_PERSON_DIR_RE = re.compile(r"^p\d+$")
_IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def _embed_crop(backend: InsightFaceBackend, img: np.ndarray):
    """Return the ArcFace embedding for the largest face in ``img``, or None.

    Saved crops are tight around the face, which can starve the detector; if the
    first pass finds nothing, retry once with a reflective border for margin.
    """
    for pad in (0.0, 0.3):
        frame = img
        if pad > 0.0:
            h, w = img.shape[:2]
            m = int(round(max(h, w) * pad))
            frame = cv2.copyMakeBorder(img, m, m, m, m, cv2.BORDER_REFLECT)
        locations, encodings = backend.detect(frame)
        if encodings:
            # Pick the largest detected face (top, right, bottom, left).
            def _area(loc):
                t, r, b, l = loc
                return (b - t) * (r - l)
            best = max(range(len(encodings)), key=lambda i: _area(locations[i]))
            return encodings[best]
    return None


def migrate(db_dir: str, det_size: int, det_thresh: float, dry_run: bool) -> int:
    if not os.path.isdir(db_dir):
        logger.error("db-dir %s does not exist", db_dir)
        return 1

    backend = InsightFaceBackend(det_size=det_size, det_thresh=det_thresh)
    db = FaceDatabase(db_dir=db_dir, metric="cosine")  # fresh, not loaded

    person_dirs = sorted(
        d for d in os.listdir(db_dir)
        if _PERSON_DIR_RE.match(d) and os.path.isdir(os.path.join(db_dir, d))
    )
    if not person_dirs:
        logger.warning("No pNNN person directories under %s -- nothing to migrate.",
                       db_dir)
        return 1

    total_people, total_embedded, total_skipped = 0, 0, 0
    for pid in person_dirs:
        pdir = os.path.join(db_dir, pid)
        images = sorted(
            f for f in os.listdir(pdir)
            if f.lower().endswith(_IMAGE_EXTS)
        )
        embedded = 0
        for fname in images:
            path = os.path.join(pdir, fname)
            img = cv2.imread(path)
            if img is None:
                logger.warning("  %s/%s: unreadable, skipping", pid, fname)
                total_skipped += 1
                continue
            enc = _embed_crop(backend, img)
            if enc is None:
                logger.warning("  %s/%s: no face detected, skipping", pid, fname)
                total_skipped += 1
                continue
            if not dry_run:
                db.add_encoding(pid, enc)
            embedded += 1
        if embedded:
            total_people += 1
            total_embedded += embedded
            logger.info("%s: %d/%d crops embedded", pid, embedded, len(images))
        else:
            logger.warning("%s: no usable crops (%d images)", pid, len(images))

    logger.info("Migrated %d people, %d embeddings (%d crops skipped).",
                total_people, total_embedded, total_skipped)
    if dry_run:
        logger.info("Dry run -- %s not written.", db.db_file)
        return 0
    if total_embedded == 0:
        logger.error("Nothing embedded; refusing to write an empty db.")
        return 1
    db.save()
    logger.info("Wrote %s (schema v%d, cosine).", db.db_file, db.schema_version)
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db-dir", default="known_faces",
                        help="Face db directory holding pNNN/ crop folders")
    parser.add_argument("--det-size", type=int, default=640,
                        help="InsightFace detector input size (square)")
    parser.add_argument("--det-thresh", type=float, default=0.5,
                        help="InsightFace minimum detector confidence")
    parser.add_argument("--dry-run", action="store_true",
                        help="Embed and report, but do not write the db")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    raise SystemExit(migrate(args.db_dir, args.det_size, args.det_thresh,
                             args.dry_run))


if __name__ == "__main__":
    main()
