# src/ui_compat.py — compat layer per Streamlit vecchio/nuovo
from __future__ import annotations
from typing import Optional
import streamlit as st

def altair_chart(chart, *, height: Optional[int] = None):
    if height is not None:
        chart = chart.properties(height=height)
    try:
        return st.altair_chart(chart, width="stretch")
    except TypeError:
        return st.altair_chart(chart, use_container_width=True)

def dataframe(data, *, height: Optional[int] = None):
    kwargs = {}
    if height is not None:
        kwargs["height"] = height
    try:
        return st.dataframe(data, width="stretch", **kwargs)
    except TypeError:
        return st.dataframe(data, use_container_width=True, **kwargs)

def line_chart(data=None, *, height: Optional[int] = None):
    kwargs = {}
    if height is not None:
        kwargs["height"] = height
    try:
        return st.line_chart(data, width="stretch", **kwargs)
    except TypeError:
        return st.line_chart(data, use_container_width=True, **kwargs)
