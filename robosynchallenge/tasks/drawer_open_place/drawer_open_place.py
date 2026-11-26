"""Compatibility imports for the native drawer_open_place task."""

from robosynchallenge.drawer_program import DrawerOpenPlace as DrawerOpenPlaceEnv

DrawerOpenPlaceTestEnv = DrawerOpenPlaceEnv
DrawerOpenPlaceAgentEnv = DrawerOpenPlaceEnv

__all__ = ['DrawerOpenPlaceEnv', 'DrawerOpenPlaceTestEnv', 'DrawerOpenPlaceAgentEnv']
