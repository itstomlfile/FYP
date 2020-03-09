import pandas as pd
import matplotlib.pyplot as plt


def traffic_preproc():
    df = pd.read_csv('data/Traffic_csv.csv',
                       usecols=['local_authority_name', 'count_date', 'all_motor_vehicles'])

    traffic_data = pd.DataFrame(columns=['local_authority_name', 'count_date', 'all_motor_vehicles'])
    for i in range(1, 1048575):
        if df['local_authority_name'][i] == 'Kirklees':
            traffic_data = traffic_data.append({'local_authority_name': df['local_authority_name'][i], 'count_date' : df['count_date'][i], 'all_motor_vehicles': df['all_motor_vehicles'][i]}, ignore_index=True)
    traffic_data['count_date'] = pd.to_datetime(traffic_data['count_date'])
    traffic_data.to_csv(r'Kirklees_Year_Only_CSV.csv', index=False)


def emissions_preproc():
    frames = []
    xls = pd.ExcelFile('data/emissions.xlsx')
    for i in range(2007, 2012):
        sheet_df = sheet_reader(str(i), xls)
        sheet_df['Start time'] = pd.to_datetime(sheet_df['Start time'])
        frames.append(sheet_df)
    emissions_data = pd.concat(frames, ignore_index=True)
    emissions_data.to_csv(r'data/emissions_csv.csv')


def sheet_reader(sheet_name, xls):
    df = pd.read_excel(xls, sheet_name=sheet_name, usecols=['Start time', 'NO2'])
    return df


def display_data():
    # Traffic Data
    traffic_data = pd.read_csv('data/Kirklees_csv.csv', usecols=['local_authority_name', 'count_date', 'all_motor_vehicles'])
    traffic_data['count_date'] = pd.to_datetime(traffic_data['count_date'])
    traffic_data.plot(x='count_date', y='all_motor_vehicles')
    plt.show()
    plt.savefig('graphs/traffic_2000-2005.png')

    # Emissions data
    emissions_data = pd.read_csv('data/emissions_csv.csv', usecols=['Start time', 'NO2'])
    emissions_data.plot(x='Start time', y='NO2')
    plt.show()
    plt.savefig('graphs/emissions_2007-2012.png')


if __name__ == '__main__':
    # emissions_preproc()
    # traffic_preproc()
    display_data()
