import pandas as pd
from pptx import Presentation
from pptx.chart.data import CategoryChartData, ChartData
from pptx.util import Inches, Pt
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.dml.color import RGBColor, MSO_THEME_COLOR
from typing import Literal
import os
from openpyxl import Workbook
import pandas as pd


def df_to_excel(df: pd.DataFrame, excel_path: str, sheet_name: str):
    if df.shape == (0, 0):
        print('Current state of dataframe is None')
    if not os.path.exists(excel_path):
        workbook = Workbook()
        sheet = workbook.active  # Truy cập vào sheet hiện tại
        sheet.title = sheet_name  # Đặt tên cho sheet
        workbook.save(excel_path)
    with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
        df.to_excel(writer, sheet_name=sheet_name)

chart_type_lookup = {
    'column': XL_CHART_TYPE.COLUMN_CLUSTERED,
    'bar': XL_CHART_TYPE.BAR_CLUSTERED,
    'pie': XL_CHART_TYPE.PIE
}

theme_color = [
    MSO_THEME_COLOR.ACCENT_1,
    MSO_THEME_COLOR.ACCENT_2,
    MSO_THEME_COLOR.ACCENT_3,
    MSO_THEME_COLOR.ACCENT_4,
    MSO_THEME_COLOR.ACCENT_5,
    MSO_THEME_COLOR.ACCENT_6,
]

def create_pptx_chart(
        template_path,
        dataframe,
        type=Literal['column', 'bar', 'pie'], 
        title=None,
    ):
    prs = Presentation(template_path) if os.path.exists(template_path) else Presentation()
    chart_type = chart_type_lookup[type]
    slide_layout = prs.slide_layouts[5]  # Layout trống
    slide = prs.slides.add_slide(slide_layout)

    if type == 'pie':
        chart_data = ChartData()
        chart_data.categories = [str(i) for i in dataframe['category']]
        chart_data.add_series('Series', dataframe['count'])
    else:
        chart_data = CategoryChartData()
        if 'category' in dataframe.columns:
            chart_data.categories = [str(i) for i in dataframe['category']]
            chart_data.add_series('count', dataframe['count'])
        elif 'row_value' in dataframe.columns:
            chart_data.categories = dataframe['row_value']
            for series_name in dataframe.columns[1:]:
                chart_data.add_series(series_name, dataframe[series_name])

    x, y, cx, cy = Inches(2), Inches(2), Inches(7), Inches(4)
    chart = slide.shapes.add_chart(
        chart_type,  # Sử dụng loại biểu đồ này
        x, y, cx, cy,
        chart_data
    ).chart

    # chart.chart_style = 2
    # chart.font.size = 101600
    chart.font.name = 'Montserrat'
    chart.has_legend = True
    chart.has_title = True
    chart.chart_title.text_frame.text = title
    chart.legend.position = XL_LEGEND_POSITION.TOP
    chart.legend.font.size = Pt(12)
    try:
        chart.category_axis.has_major_gridlines = False
        chart.category_axis.has_minor_gridlines = False
        chart.category_axis.has_title = False
        chart.category_axis.visible = True
        chart.category_axis.tick_labels.font.size = Pt(10)
    except:
        pass
    try:
        chart.value_axis.has_major_gridlines = False
        chart.value_axis.has_minor_gridlines = False
        chart.value_axis.visible = False
    except:
        pass

    if type != 'pie':
        for index, i in enumerate(chart.series):
            i.data_labels.font.size = Pt(8)
            i.data_labels.font.name = 'Montserrat'
            i.data_labels.number_format = 'General'
            i.data_labels.number_format_is_linked = True
            i.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
            i.data_labels.show_category_name = False
            i.data_labels.show_legend_key = False
            i.data_labels.show_percentage = False
            i.data_labels.show_series_name = False
            i.data_labels.show_value = True
            i.format.fill.solid()
            try:
                i.format.fill.fore_color.theme_color = theme_color[index]
            except:
                i.format.fill.fore_color.theme_color = theme_color[0]
    else:
        for serie in chart.series:
            for idx, point in enumerate(serie.points):
                point.format.fill.solid()
                try:
                    point.format.fill.fore_color.theme_color = theme_color[idx]
                except:
                    point.format.fill.fore_color.theme_color = theme_color[0]

    prs.save(template_path)

# def create_pptx_chart_old(
#         ppt_path, 
#         dataframe, 
#         type=Literal['column', 'bar', 'pie'], 
#         position=[2,2,7,4],
#         title=None,
#         background_image_path=None,
#         use_template=True
#     ):
        

#     prs = Presentation(ppt_path) if os.path.exists(ppt_path) else Presentation()
#     chart_type = chart_type_lookup[type]

#     slide_layout = prs.slide_layouts[5]  # Layout trống
#     slide = prs.slides.add_slide(slide_layout)

#     if background_image_path:
#         slide.shapes.add_picture(background_image_path, 0, 0, width=prs.slide_width, height=prs.slide_height)


#     chart_data = CategoryChartData()


#     if 'category' in dataframe.columns:
#         chart_data.categories = dataframe['category']
#         chart_data.add_series('count', dataframe['count'])
#     elif 'row_value' in dataframe.columns:
#         chart_data.categories = dataframe['row_value']
#         for series_name in dataframe.columns[1:]:
#             chart_data.add_series(series_name, dataframe[series_name])

#     x, y, cx, cy = Inches(position[0]), Inches(position[1]), Inches(position[2]), Inches(position[3])
#     chart = slide.shapes.add_chart(
#         chart_type,  # Sử dụng loại biểu đồ này
#         x, y, cx, cy,
#         chart_data
#     ).chart

#     if use_template:
#         for i in prs.slides[0].shapes:
#             if hasattr(i, 'chart'):
#                 chart_template = i.chart
#             # print('No chart')
#         # chart_template = prs.slides[0].shapes[0].chart
#         # chart_template = slide.shapes[0].chart
#         chart.chart_style = chart_template.chart_style  # Kiểu dáng của biểu đồ
#         chart.has_legend = chart_template.has_legend
#         chart.legend.position = chart_template.legend.position
#         # Sao chép màu sắc cho các series trong biểu đồ
#         for i, series in enumerate(chart.series):
#             try:
#                 template_series = chart_template.series[i]
#                 fill = series.format.fill
#                 template_fill = template_series.format.fill
#                 if template_fill.type == 'solid':
#                     fill.solid()
#                     fill.fore_color.rgb = template_fill.fore_color.rgb
#             except:
#                 pass
         

#         # Sao chép cỡ chữ cho các data labels, legend, và axis labels
#         for i, series in enumerate(chart.series):
#             try:
#                 template_series = chart_template.series[i]
#                 series.has_data_labels = template_series.has_data_labels
#                 if series.has_data_labels:
#                     data_labels = series.data_labels
#                     template_data_labels = template_series.data_labels
#                     data_labels.font.size = template_data_labels.font.size
#                     data_labels.font.bold = template_data_labels.font.bold
#                     data_labels.font.color.rgb = template_data_labels.font.color.rgb
#             except:
#                 pass

#         # Sao chép cỡ chữ cho legend (chú giải)
#         legend = chart.legend
#         legend.font.size = chart_template.legend.font.size
#         legend.font.bold = chart_template.legend.font.bold
#         # legend.font.color.rgb = chart_template.legend.font.color.rgb

#         # Sao chép cỡ chữ cho axis labels
#         category_axis = chart.category_axis
#         template_category_axis = chart_template.category_axis
#         category_axis.has_title = template_category_axis.has_title
#         if category_axis.has_title:
#             category_axis.axis_title.text_frame.text = template_category_axis.axis_title.text_frame.text
#             category_axis.axis_title.text_frame.paragraphs[0].font.size = template_category_axis.axis_title.text_frame.paragraphs[0].font.size

#         value_axis = chart.value_axis
#         template_value_axis = chart_template.value_axis
#         value_axis.has_title = template_value_axis.has_title
#         if value_axis.has_title:
#             value_axis.axis_title.text_frame.text = template_value_axis.axis_title.text_frame.text
#             value_axis.axis_title.text_frame.paragraphs[0].font.size = template_value_axis.axis_title.text_frame.paragraphs[0].font.size



#     if title:
#         chart.has_title = True
#         chart.chart_title.text_frame.text = title

#     # chart.has_legend = True
#     # chart.legend.position = XL_LEGEND_POSITION.TOP  # Vị trí của legend
    
#     # for series in chart.series:
#     #     series.has_data_labels = True
#     #     data_labels = series.data_labels
#     #     data_labels.number_format = '0'  # Định dạng số
#     #     data_labels.position = XL_LABEL_POSITION.OUTSIDE_END  # Vị trí của data labels
#     #     data_labels.show_legend_key = False  # Ẩn legend key nếu có
#     #     data_labels.show_category_name = False  # Ẩn tên category nếu có
#     #     data_labels.show_series_name = False  # Ẩn tên series nếu có
#     #     data_labels.show_percentage = False  # Ẩn phần trăm nếu có
#     #     data_labels.show_value = True  # Hiển thị giá trị

#     prs.save(ppt_path)


