"""
XSD validation of an exported XPlanGML 6.1 document, fully offline.

The entry schema is XPlanung-Operationen.xsd (it defines the XPlanAuszug
root, everything else is included from there). Two schemas are imported by
absolute w3.org URL -- xlink.xsd (from the GML profile) and xml.xsd (from
xlink.xsd) -- and are redirected to local copies under schemas/ instead of
being fetched on every validation. Network access is disabled, so a missing
local copy fails loudly instead of silently going online.

This only checks XSD conformance (structure, types, element order). The
XPlanValidator's semantic and geometric rules are a separate step.
"""

import functools
import os

from lxml import etree

_SCHEMAS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schemas")
ENTRY_SCHEMA = os.path.join(_SCHEMAS_DIR, "xplanung", "6.1", "XPlanung-Operationen.xsd")

_LOCAL_COPIES = {
    "http://www.w3.org/1999/xlink.xsd": os.path.join(_SCHEMAS_DIR, "xlink", "xlink.xsd"),
    "http://www.w3.org/2001/xml.xsd": os.path.join(_SCHEMAS_DIR, "xml", "xml.xsd"),
}


class _LocalSchemaResolver(etree.Resolver):
    def resolve(self, url, pubid, context):
        local = _LOCAL_COPIES.get(url)
        if local is None:
            return None  # fall through to lxml's default (relative paths inside schemas/)
        return self.resolve_filename(local, context)


@functools.lru_cache(maxsize=1)
def _schema():
    """Parsing the full XPlanGML schema set takes a moment -- do it once per process."""
    parser = etree.XMLParser(no_network=True)
    parser.resolvers.add(_LocalSchemaResolver())
    return etree.XMLSchema(etree.parse(ENTRY_SCHEMA, parser))


def validate(document):
    """
    document : path to a .gml file, or an lxml element/tree (e.g. straight
        from serialize.build()).

    Returns a list of {"line", "message"} dicts -- empty if the document is
    XSD-valid.
    """
    if isinstance(document, (str, os.PathLike)):
        document = etree.parse(os.fspath(document))
    schema = _schema()
    if schema.validate(document):
        return []
    return [{"line": error.line, "message": error.message} for error in schema.error_log]
