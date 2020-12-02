import pandas as pd
import datetime

from insolver.frame import InsolverDataFrame


class InsuranceDFCreator:

    def __init__(self,
                 df_policies=None,
                 column_policies_policy_id=None,
                 column_policies_date_start=None,
                 column_policies_date_end=None,
                 column_policies_exposure=None,
                 column_policies_claims_count=None,
                 column_policies_claims_count_adj=None,
                 df_claims=None,
                 column_claims_policy_id=None,
                 column_claims_date_event=None,
                 column_claims_count=None,
                 column_claims_sum=None,
                 column_claims_sum_infl=None,
                 claim_date_event_max=None,
                 apply_adjust_claims_count=False,
                 apply_adjust_inflation=False,
                 calendar=None,
                 column_calendar_ym=None,
                 column_calendar_k_infl=None):

        self._df_policies = df_policies
        self.column_policies_policy_id = column_policies_policy_id
        self.column_policies_date_start = column_policies_date_start
        self.column_policies_date_end = column_policies_date_end
        self.column_policies_exposure = column_policies_exposure
        self.column_policies_claims_count = column_policies_claims_count
        self.column_policies_claims_count_adj = column_policies_claims_count_adj
        self._df_claims = df_claims
        self.column_claims_policy_id = column_claims_policy_id
        self.column_claims_date_event = column_claims_date_event
        self.column_claims_count = column_claims_count
        self.column_claims_sum = column_claims_sum
        self.column_claims_sum_infl = column_claims_sum_infl
        self.columns_claims_to_sum = []
        self.claim_date_event_max = claim_date_event_max
        self.apply_adjust_claims_count = apply_adjust_claims_count
        self.apply_adjust_inflation = apply_adjust_inflation
        self._calendar = calendar
        self.column_calendar_ym = column_calendar_ym
        self.column_calendar_k_infl = column_calendar_k_infl

        self._check_claims_df()

        if isinstance(self._df_policies, pd.DataFrame) and isinstance(self._df_claims, pd.DataFrame):
            if not (isinstance(self.column_policies_policy_id, str) and isinstance(self.column_claims_policy_id, str)):
                raise NotImplementedError("'column_policies_policy_id' and 'column_claims_policy_id' should be defined.")
            self._df_policies_result = self._df_policies.merge(self._get_claims_grouped_df,
                                                               how='left',
                                                               left_on=self.column_policies_policy_id,
                                                               right_on=self.column_claims_policy_id,
                                                               suffixes=('_pol', '_cl'))
            if self.column_claims_count in list(self._df_policies_result.columns):
                self.column_policies_claims_count = self.column_claims_count
            elif self.column_claims_count + '_cl' in list(self._df_policies_result.columns):
                self.column_policies_claims_count = self.column_claims_count + '_cl'
            self._df_claims_result = self._df_claims.merge(self._df_policies,
                                                           how='left',
                                                           left_on=self.column_claims_policy_id,
                                                           right_on=self.column_policies_policy_id,
                                                           suffixes=('_cl', '_pol'))
        elif isinstance(self._df_policies, pd.DataFrame):
            self._df_policies_result = self._df_policies.copy()
        elif isinstance(self._df_claims, pd.DataFrame):
            if isinstance(self.column_claims_date_event, str):
                self._df_claims_result = self._df_claims[self._df_claims[self.column_claims_date_event] <=
                                                         self.claim_date_event_max].copy()
            else:
                self._df_claims_result = self._df_claims.copy()
        else:
            raise NotImplementedError("No DataFrame is defined.")

        self._check_policies_df()

    def get_policies(self):
        if self._df_policies_result:
            return InsolverDataFrame(self._df_policies_result)
        else:
            return None

    def get_claims(self):
        if self._df_claims_result:
            return InsolverDataFrame(self._df_claims_result)
        else:
            return None

    def _check_claims_df(self):
        if isinstance(self._df_claims, pd.DataFrame):
            if not isinstance(self.column_claims_sum, str):
                raise NotImplementedError("'df_claims' should contain 'column_claims_sum'.")
            if not isinstance(self.claim_date_event_max, datetime.datetime):
                self.claim_date_event_max = datetime.datetime.now().date()
            if not isinstance(self.column_claims_count, str):
                self.column_claims_count = 'claim_count'
                while self.column_claims_count in list(self._df_claims.columns):
                    self.column_claims_count = self.column_claims_count + '_'
                self._df_claims[self.column_claims_count] = 1
            self.columns_claims_to_sum.append(self.column_claims_count)
            if self.apply_adjust_inflation:
                if not (isinstance(self._calendar, pd.DataFrame) and isinstance(self.column_calendar_ym, str) and
                        isinstance(self.column_calendar_k_infl, str) and isinstance(self.column_claims_date_event, str)
                        and isinstance(self.column_claims_sum_infl, str)):
                    raise NotImplementedError("'calendar', 'column_calendar_ym', 'column_calendar_k_infl' and"
                                              " 'column_claims_date_event', 'column_claims_sum_infl' should be"
                                              " defined if 'apply_adjust_inflation' is True.")
                else:
                    self.column_claims_date_event_ym = 'claim_date_event_ym'
                    while self.column_claims_date_event_ym in list(self._df_claims.columns):
                        self.column_claims_date_event_ym = self.column_claims_date_event_ym + '_'
                    self._df_claims[self.column_claims_date_event_ym] = self._df_claims[self.column_claims_date_event].\
                        apply(self._get_ym)
                    self._df_claims = self._df_claims.merge(self._calendar
                                                            [[self.column_calendar_ym, self.column_calendar_k_infl]],
                                                            how='left',
                                                            left_on=self.column_claims_date_event_ym,
                                                            right_on=self.column_calendar_ym,
                                                            suffixes=('_cl', '_cal'))
                    if self.column_calendar_k_infl in list(self._df_claims.columns):
                        self.column_claims_k_infl = self.column_calendar_k_infl
                    elif self.column_calendar_k_infl + '_cal' in list(self._df_claims.columns):
                        self.column_claims_k_infl = self.column_calendar_k_infl + '_cal'
                    self._df_claims[self.column_claims_sum_infl] = self._df_claims[self.column_claims_sum] * \
                                                                   self._df_claims[self.column_claims_k_infl]
                    self.columns_claims_to_sum.append(self.column_claims_sum_infl)
            else:
                self.columns_claims_to_sum.append(self.column_claims_sum)

    def _get_claims_grouped_df(self):
        if isinstance(self.column_claims_date_event, str):
            return self._df_claims \
                [self._df_claims[self.column_claims_date_event] <= self.claim_date_event_max] \
                [[self.column_claims_policy_id] + self.columns_claims_to_sum]. \
                groupby(by=self.column_claims_policy_id, as_index=False).sum()
        else:
            return self._df_claims \
                [[self.column_claims_policy_id] + self.columns_claims_to_sum]. \
                groupby(by=self.column_claims_policy_id, as_index=False).sum()

    def _check_policies_df(self):
        if isinstance(self._df_policies_result, pd.DataFrame):
            if not isinstance(self.column_policies_claims_count, str):
                raise NotImplementedError("'_df_policies_result' should contain 'column_policies_claims_count'.")
            if self.apply_adjust_claims_count:
                if not (isinstance(self.column_policies_date_start, str) and
                        isinstance(self.column_policies_date_end, str)):
                    raise NotImplementedError("'column_policies_date_start' and 'column_policies_date_end'"
                                              " should be defined if 'apply_adjust_exposure' is True.")
                if not isinstance(self.column_policies_exposure, str):
                    self.column_policies_exposure = 'exposure'
                    while self.column_policies_exposure in list(self._df_policies_result.columns):
                        self.column_policies_exposure = self.column_policies_exposure + '_'
                    self._df_policies_result[self.column_policies_exposure] = self._df_policies_result \
                        [[self.column_policies_date_start, self.column_policies_date_end]]. \
                        apply(self._get_exposure, axis=1, args=(self.claim_date_event_max,))
                if not isinstance(self.column_policies_claims_count_adj, str):
                    self.column_policies_claims_count_adj = 'claims_count_adj'
                    while self.column_policies_claims_count_adj in list(self._df_policies_result.columns):
                        self.column_policies_claims_count_adj = self.column_policies_claims_count_adj + '_'
                self._df_policies_result[self.column_policies_claims_count_adj] = self._df_policies_result \
                    [[self.column_policies_exposure, self.column_policies_claims_count]]. \
                    apply(self._get_claims_count_adj, axis=1)

    @staticmethod
    def _get_exposure(date_start_end, claim_date_event_max):
        policy_date_start = date_start_end[0]
        policy_date_end = date_start_end[1]
        if policy_date_end > claim_date_event_max:
            exposure = ((claim_date_event_max - policy_date_start).days + 1) / 365
        elif (policy_date_end - policy_date_start).days < 364:
            exposure = ((policy_date_end - policy_date_start).days + 1) / 365
        else:
            exposure = 1
        return exposure

    @staticmethod
    def _get_claims_count_adj(exposure_claims_count):
        exposure = exposure_claims_count[0]
        claims_count = exposure_claims_count[1]
        return claims_count / exposure

    @staticmethod
    def _get_ym(date):
        if pd.isnull(date):
            return None
        elif date.month < 10:
            return str(date.year) + '-0' + str(date.month)
        else:
            return str(date.year) + '-' + str(date.month)
