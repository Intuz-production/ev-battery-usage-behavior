"""
Pages module for EV Battery Analysis
"""
from ._upload_page import render_upload_page
from ._soh_page import render_soh_page
from ._rul_page import render_rul_page

__all__ = ['render_upload_page', 'render_soh_page', 'render_rul_page']