#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:56:28 2026

@author: felipemoret
"""

# -*- coding: utf-8 -*-
"""
@author: Adele Ravagnani

Module that defines a class to load and clean a LOBSTER data set.
This is made up of a "message file" and an "order book file" and it is related to a given asset and a given trading day.

Reference to: https://lobsterdata.com, Bouchaud, J.-P., Bonart, J., Donier, J., & Gould, M. (2018). Trades, quotes and prices: Financial markets under the microscope. Cambridge University Press.
"""

import warnings
import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
from pathlib import Path

#%%
class LOB_data:
    def __init__(self, path_folder, label_message_file, label_ob_file, tick_size,
                 auto_add_day_time_cols: bool = True,
                 dayid_col: str = "DayID",
                 timeabs_col: str = "TimeAbs"):
        super().__init__()
        self.path_folder = path_folder
        self.label_message_file = label_message_file
        self.label_ob_file = label_ob_file
        self.message_file = None
        self.ob_file = None
        self.n_levels = None
        self.tick_size = tick_size

        # --- multi-day calibration helpers ---
        self.auto_add_day_time_cols = bool(auto_add_day_time_cols)
        self.dayid_col = str(dayid_col)
        self.timeabs_col = str(timeabs_col)

    # ------------------------------------------------------------------
    # Internal helpers (multi-day safe)
    # ------------------------------------------------------------------
    def _ensure_aligned_index(self) -> None:
        """
        Keep message_file and ob_file perfectly aligned with a clean RangeIndex.
        IMPORTANT when concatenating multiple days: indexes may be non-unique.
        """
        if self.message_file is None or self.ob_file is None:
            return

        if len(self.message_file) != len(self.ob_file):
            raise ValueError(
                f"message_file and ob_file must have same length. "
                f"Got {len(self.message_file)} vs {len(self.ob_file)}."
            )

        self.message_file = self.message_file.reset_index(drop=True)
        self.ob_file = self.ob_file.reset_index(drop=True)

    def _compute_day_id(self) -> pd.Series:
        """
        Detect day boundaries when Time resets (diff < 0). Returns a 0..K DayID series.
        Works for concatenated multi-day LOBSTER where Time is seconds after midnight.
        """
        if self.message_file is None or len(self.message_file) == 0:
            return pd.Series([], dtype=int)

        t = pd.to_numeric(self.message_file["Time"], errors="coerce").astype(float)
        day_id = (t.diff() < 0).fillna(False).cumsum().astype(int)
        return day_id

    def _refresh_day_time_cols(self, force: bool = False) -> None:
        """
        Ensure DayID and TimeAbs exist and are consistent.

        - DayID: 0,1,2,... according to detected resets in Time.
        - TimeAbs: monotone "stitched time" in seconds: Time + 86400*DayID.
        """
        if not self.auto_add_day_time_cols:
            return
        if self.message_file is None or len(self.message_file) == 0:
            return

        need_day = force or (self.dayid_col not in self.message_file.columns)
        need_abs = force or (self.timeabs_col not in self.message_file.columns)

        # If both already exist and we are not forcing, do nothing (fast path)
        if (not need_day) and (not need_abs):
            return

        day_id = self._compute_day_id()
        if need_day:
            self.message_file[self.dayid_col] = day_id

        if need_abs:
            t = pd.to_numeric(self.message_file["Time"], errors="coerce").astype(float)
            self.message_file[self.timeabs_col] = t + 86400.0 * day_id.astype(float)

    # ------------------------------------------------------------------
    # Construction / loading
    # ------------------------------------------------------------------
    def build_class_if_files_already_loaded(self, message_file, ob_file):
        """
        Build the class from already loaded dataframes (e.g., concatenated multi-day).
        PRO-CALIBRATION: auto-creates DayID and TimeAbs right here.
        """
        self.message_file = message_file
        self.ob_file = ob_file
        self.n_levels = len(self.ob_file.columns) // 4

        self._ensure_aligned_index()
        # <-- requested behavior: always add DayID/TimeAbs after build_class...
        self._refresh_day_time_cols(force=True)
        return

    def load_LOB_data(self, verbose=True, message_file_with_extra_column=False):
        """
        Loads ONE pair of LOBSTER files.
        (For multi-day, you typically concat outside and call build_class_if_files_already_loaded.)
        """
        msg_path = Path(self.path_folder).expanduser() / self.label_message_file
        ob_path = Path(self.path_folder).expanduser() / self.label_ob_file

        self.message_file = pd.read_csv(msg_path, header=None, low_memory=False)
        if message_file_with_extra_column is True:
            self.message_file = self.message_file.iloc[:, :-1]

        self.message_file.columns = ["Time", "Type", "ID", "Size", "Price", "Direction"]
        self.message_file["Price"] = self.message_file["Price"] / 10000.0

        self.ob_file = pd.read_csv(ob_path, header=None)
        self.n_levels = len(self.ob_file.columns) // 4

        header_ob_file = []
        for k in range(self.n_levels):
            header_ob_file.append("AskPrice_" + str(k + 1))
            header_ob_file.append("AskSize_" + str(k + 1))
            header_ob_file.append("BidPrice_" + str(k + 1))
            header_ob_file.append("BidSize_" + str(k + 1))
        self.ob_file.columns = header_ob_file

        for k in range(self.n_levels):
            self.ob_file["AskPrice_" + str(k + 1)] = self.ob_file["AskPrice_" + str(k + 1)] / 10000.0
            self.ob_file["BidPrice_" + str(k + 1)] = self.ob_file["BidPrice_" + str(k + 1)] / 10000.0

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("Data set length:", len(self.ob_file))
            print("\nMessage file first lines")
            print(self.message_file.head())
            print("\nOrder book file first lines")
            print(self.ob_file.head())

        return

    # ------------------------------------------------------------------
    # Cleaning steps (multi-day safe)
    # ------------------------------------------------------------------
    def clean_trading_halts(self, verbose=True):
        """
        Multi-day safe trading halt removal.
        Handles multiple halts, and pairs start/end events *within each day segment*.
        Drops by index positions (not by Time interval), avoiding cross-day mistakes.
        """
        if self.message_file is None or len(self.message_file) == 0:
            return

        if 7 not in self.message_file["Type"].drop_duplicates().tolist():
            if verbose is True:
                print("No trading halt messages found.")
            return

        self._ensure_aligned_index()

        day_id = self._compute_day_id().to_numpy()
        typ = self.message_file["Type"].to_numpy()
        direction = self.message_file["Direction"].to_numpy()
        price = self.message_file["Price"].to_numpy()

        # Warn if type-7 events exist with unexpected direction values
        type7_mask = typ == 7
        unexpected_dirs = direction[type7_mask]
        if np.any(unexpected_dirs != -1):
            unique_dirs = np.unique(unexpected_dirs[unexpected_dirs != -1])
            warnings.warn(
                f"Type-7 (trading halt) events found with unexpected direction values: "
                f"{unique_dirs.tolist()}. Only direction == -1 is handled; others will be ignored."
            )

        to_drop_pos = []

        for d in np.unique(day_id):
            idx_d = np.where((day_id == d) & (typ == 7) & (direction == -1))[0]
            if idx_d.size == 0:
                continue

            start_pos = None
            for pos in idx_d:
                if price[pos] == -1:
                    start_pos = pos
                elif price[pos] == 1:
                    if start_pos is not None and pos >= start_pos:
                        to_drop_pos.extend(range(start_pos, pos + 1))
                        start_pos = None
                    else:
                        to_drop_pos.append(pos)

            if start_pos is not None:
                to_drop_pos.append(start_pos)

        if len(to_drop_pos) > 0:
            to_drop_pos = np.unique(np.asarray(to_drop_pos, dtype=int))
            drop_idx = self.message_file.index[to_drop_pos]
            self.message_file.drop(drop_idx, inplace=True)
            self.ob_file.drop(drop_idx, inplace=True)

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Trading halt found and handled.")
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("Data set length:", len(self.ob_file))

        return

    def clean_opening_closing_auctions(self, verbose=True):
        ind_opening_auction = self.message_file[(self.message_file["Type"] == 6) & (self.message_file["ID"] == -1)].index
        ind_closing_auction = self.message_file[(self.message_file["Type"] == 6) & (self.message_file["ID"] == -2)].index

        if len(ind_opening_auction) > 0:
            self.ob_file.drop(ind_opening_auction, inplace=True)
            self.message_file.drop(ind_opening_auction, inplace=True)
        if len(ind_closing_auction) > 0:
            self.ob_file.drop(ind_closing_auction, inplace=True)
            self.message_file.drop(ind_closing_auction, inplace=True)

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("Data set length:", len(self.ob_file))

        return

    def clean_crossed_prices_obs(self, verbose=True):
        ind_to_drop = self.ob_file[self.ob_file["AskPrice_1"] <= self.ob_file["BidPrice_1"]].index
        if len(ind_to_drop) > 0:
            self.ob_file.drop(ind_to_drop, inplace=True)
            self.message_file.drop(ind_to_drop, inplace=True)

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("Data set length:", len(self.ob_file))

        return

    def handle_splitted_lo_executions(self, verbose=True):
        """
        Multi-day safe split-execution handling.

        Fix:
        - DO NOT groupby('Time'), because Time repeats across days.
        - Merge only *consecutive* events with identical Time.
        """
        if self.message_file is None or len(self.message_file) == 0:
            return

        self._ensure_aligned_index()
        self._refresh_day_time_cols()  # Garante que a coluna DayID exista

        t = pd.to_numeric(self.message_file["Time"], errors="coerce").astype(float).to_numpy()
        day_id = self.message_file[self.dayid_col].to_numpy(dtype=int)
        n = len(t)
        if n < 2:
            return

        if not np.any(t[1:] == t[:-1]):
            if verbose is True:
                print(f"Out of {n} events, {n} are associated to unique times (consecutive).")
            return

        # Uma sequência de timestamps idênticos termina quando o 'Time' muda OU o 'DayID' muda.
        time_changed = t[1:] != t[:-1]
        day_changed = day_id[1:] != day_id[:-1]
        run_ends_mask = time_changed | day_changed

        if not np.any(~run_ends_mask):  # Nenhum par consecutivo com (time, day) idênticos
            if verbose is True:
                print(f"Out of {n} events, {n} are associated to unique times (consecutive).")
            return

        # Identifica corretamente os inícios de sequências de (time, day) idênticos
        run_starts = np.r_[0, np.where(run_ends_mask)[0] + 1]
        run_ends = np.r_[run_starts[1:], n]
        run_len = run_ends - run_starts
        multi_runs = np.where(run_len > 1)[0]

        if verbose is True:
            print(f"Out of {n} events, {len(run_starts)} are associated to unique consecutive times (runs).")

        typ = self.message_file["Type"].to_numpy()
        direction = self.message_file["Direction"].to_numpy()
        size = pd.to_numeric(self.message_file["Size"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
        price = pd.to_numeric(self.message_file["Price"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

        drop_positions = []
        col_size = self.message_file.columns.get_loc("Size")
        col_price = self.message_file.columns.get_loc("Price")

        for r in multi_runs:
            s = int(run_starts[r])
            e = int(run_ends[r])  # exclusive

            if typ[s] != 4:
                continue
            if not np.all(typ[s:e] == 4):
                continue

            if not np.all(direction[s:e] == direction[s]):
                continue

            total_size = float(size[s:e].sum())
            if total_size <= 0:
                continue

            wavg_price = float(np.sum(price[s:e] * size[s:e]) / total_size)

            last = e - 1
            self.message_file.iat[last, col_size] = total_size
            self.message_file.iat[last, col_price] = wavg_price

            if last > s:
                drop_positions.extend(range(s, last))

        if len(drop_positions) > 0:
            drop_positions = np.unique(np.asarray(drop_positions, dtype=int))
            drop_idx = self.message_file.index[drop_positions]
            self.message_file.drop(drop_idx, inplace=True)
            self.ob_file.drop(drop_idx, inplace=True)

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("New shape is", len(self.ob_file))

        return

    def handle_hidden_orders(self, flag_drop=True, verbose=True):
        """
        Vectorized hidden order handling (faster for large datasets).
        """
        if self.message_file is None or len(self.message_file) == 0:
            return

        self._ensure_aligned_index()

        hidden_mask = (self.message_file["Type"] == 5)
        hidden_pos = np.where(hidden_mask.to_numpy())[0]

        if hidden_pos.size == 0:
            if verbose is True:
                print("No hidden orders found.")
            return

        if flag_drop is True:
            print("Dropping hidden orders ...")
            drop_idx = self.message_file.index[hidden_pos]
            self.message_file.drop(drop_idx, inplace=True)
            self.ob_file.drop(drop_idx, inplace=True)
            self._ensure_aligned_index()
        else:
            print("Reassigning directions to hidden orders ...")
            hidden_pos = hidden_pos[hidden_pos > 0]
            if hidden_pos.size > 0:
                prev_pos = hidden_pos - 1

                best_ask_prev = pd.to_numeric(self.ob_file.iloc[prev_pos]["AskPrice_1"], errors="coerce").to_numpy(dtype=float)
                best_bid_prev = pd.to_numeric(self.ob_file.iloc[prev_pos]["BidPrice_1"], errors="coerce").to_numpy(dtype=float)
                mid_prev = 0.5 * (best_bid_prev + best_ask_prev)

                order_price = pd.to_numeric(self.message_file.iloc[hidden_pos]["Price"], errors="coerce").to_numpy(dtype=float)
                new_dir = np.where(order_price <= mid_prev, +1, -1)
                self.message_file.iloc[hidden_pos, self.message_file.columns.get_loc("Direction")] = new_dir

            print("Changing labels to hidden orders type such that they are of type 4 as MOs ...")
            self.message_file["Type"] = self.message_file["Type"].replace(5, 4)

        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("New shape is", len(self.ob_file))

        return

    def clean_LOB_data(self, flag_drop_hidden_orders=True, verbose=True):
        print("\nCleaning from trading halts ...")
        self.clean_trading_halts(verbose)
        print("\nCleaning from auctions ...")
        self.clean_opening_closing_auctions(verbose)
        print("\nCleaning from crossed price observations ...")
        self.clean_crossed_prices_obs(verbose)
        print("\nHandling splitted LO executions ...")
        self.handle_splitted_lo_executions(verbose)
        print("\nHandling hidden orders ...")
        self.handle_hidden_orders(flag_drop_hidden_orders, verbose)

    def load_and_clean_LOB_data(self, flag_drop_hidden_orders=True, verbose=True,
                               message_file_with_extra_column=False, load=True):
        if load is True:
            print("\nLoading message and order book file ...")
            self.load_LOB_data(verbose, message_file_with_extra_column)
        print("\nCleaning message and order book file ...")
        self.clean_LOB_data(flag_drop_hidden_orders, verbose)
        print("\nLoading and cleaning of the dataset completed!")
        return

    def cut_before_and_after_LOB_data(self, minutes_to_cut_beginning, minutes_to_cut_end, verbose=True):
        """
        Multi-day safe cutting: applies the cut *within each detected day segment*.
        """
        if self.message_file is None or len(self.message_file) == 0:
            return

        self._ensure_aligned_index()

        t = pd.to_numeric(self.message_file["Time"], errors="coerce").astype(float)
        day_id = self._compute_day_id()

        t_first = t.groupby(day_id).transform("first")
        t_last = t.groupby(day_id).transform("last")

        t_start_new = t_first + 60.0 * float(minutes_to_cut_beginning)
        t_end_new = t_last - 60.0 * float(minutes_to_cut_end)

        keep = (t >= t_start_new) & (t <= t_end_new)

        self.message_file = self.message_file.loc[keep].copy()
        self.ob_file = self.ob_file.loc[keep].copy()

        self._ensure_aligned_index()
        self._refresh_day_time_cols(force=True)

        if verbose is True:
            print("Applied cut per day.")
            print("Check shapes of the new files:", len(self.ob_file) == len(self.message_file))
            print("New shape is", len(self.ob_file))

        return

    def save_cleaned_and_cut_files(self, path_save):
        import os
        self.message_file.to_csv(os.path.join(path_save, "cleaned_" + self.label_message_file), index=False)
        self.ob_file.to_csv(os.path.join(path_save, "cleaned_" + self.label_ob_file), index=False)
        return

    def obtain_times_datetime_format(self, day, infer_days: bool = True):
        """
        Vectorized conversion of Time into datetime.

        - For multi-day concatenated data: pass FIRST day's date in 'day' and infer_days=True
          so each detected day boundary increments +1 day in calendar.
        """
        if self.message_file is None or len(self.message_file) == 0:
            return

        self._ensure_aligned_index()

        t = pd.to_numeric(self.message_file["Time"], errors="coerce").astype(float).to_numpy(dtype=float)
        if isinstance(day, datetime.date) and not isinstance(day, datetime.datetime):
            day = datetime.datetime(day.year, day.month, day.day)
        base = float(day.timestamp())

        if infer_days:
            day_id = self._compute_day_id().to_numpy(dtype=float)
            ts = base + 86400.0 * day_id + t
        else:
            ts = base + t

        self.message_file["TimeDatetime"] = pd.to_datetime(ts, unit="s", origin="unix")
        return

    def obtain_ob_time_step(self, time_step):
        if "TimeDatetime" in self.message_file.columns:
            time_step_datetime = datetime.timedelta(seconds=time_step)

            ob_file_copy = self.ob_file.copy()
            ob_file_copy.index = self.message_file["TimeDatetime"]
            ob_file_time_step = ob_file_copy.resample(time_step_datetime).apply(
                lambda x: x.iloc[-1] if len(x) > 0 else pd.Series(np.nan, index=x.columns)
            )

            ob_file_time_step = pd.DataFrame(ob_file_time_step.values, columns=self.ob_file.columns)
            return ob_file_time_step.apply(pd.to_numeric)
        else:
            raise Exception(
                'Before applying this function, the "message file" must have a column with the times in datetime format. '
                'So, please apply the function "obtain_times_datetime_format" and then, this function.'
            )
