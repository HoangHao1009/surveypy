from ..question import MultipleAnswer, SingleAnswer, Number, Rank
from ...utils import ChartConfig
from pydantic import BaseModel
from typing import Union, List, Literal
import plotly.express as px
from plotly.subplots import make_subplots
import itertools

BaseType = Union[SingleAnswer, MultipleAnswer, Rank]
QuestionType = Union[SingleAnswer, MultipleAnswer, Rank, Number]

class Chart(BaseModel):
    base: BaseType
    target: QuestionType
    deep_by: List[BaseType] = []
    config: ChartConfig = ChartConfig()

    def _info(self):
        ctab = self.base | self.target
        ctab.config.round_perc = False
        ctab.config.perc = self.config.perc
        ctab.config.deep_by = self.deep_by
        df = ctab.dataframe
        result = {}
        deep_repsonses = [[i.value for i in deep.responses] for deep in self.deep_by]
        pairs = list(itertools.product(*deep_repsonses))
        for index, pair in enumerate(pairs):
            key = 'x'.join(pair)
            data = df.loc[:, pair].reset_index().melt(id_vars=['target_root', 'target_answer'])
            if self.config.perc:
                data['value'] = data['value'].map(lambda x: round(x, 2))

            if self.config.x_in_base:
                x = 'variable_1'
                y = 'value'
                color = 'target_answer'
            else:
                x = 'target_answer'
                y = 'value'
                color = 'variable_1'
            if self.config.orientation == 'h':
                x, y, color = y, x, color

            if self.config.chart_type == 'bar':
                chart = px.bar(data, x=x, y=y, color=color, text='value')
            elif self.config.chart_type == 'line':
                chart = px.line(data, x=x, y=y, color=color, text='value')
            result[key] = {}
            result[key]['data'] = data
            result[key]['index'] = index
            result[key]['chart'] = chart
        return result

    def show_grid(self, **kwargs):
        chart_data = self._info()
        titles = list(chart_data.keys())
        rows, cols = self.config.grid[0], self.config.grid[1]
        fig = make_subplots(rows=rows, cols=cols, subplot_titles=titles, shared_xaxes=self.config.shared_xaxes, shared_yaxes=self.config.shared_yaxes)

        for key, values in chart_data.items():
            index = values['index']
            data = values['data']
            px_chart = values['chart']
            row = (index // cols) + 1
            col = (index % cols) + 1
            for trace in px_chart.data:
                trace.showlegend = True if index == 0 else False
                fig.add_trace(trace, row=row, col=col)
        fig.update_layout(title_text=f"{self.base.code} x {self.target.code}", 
                          showlegend=True, barmode=self.config.barmode, 
                          height=self.config.height, width=self.config.width,
                          **kwargs)
        fig.show()