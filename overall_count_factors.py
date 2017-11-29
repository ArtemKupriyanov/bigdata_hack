weekday_count = pd.DataFrame(train_data.groupby(['weekday_key']).size())
weekday_count.columns = ['weekday_overall_offers']
weekday_count['weekday_key'] = weekday_count.index
hour_count = pd.DataFrame(train_data.groupby(['hour_key']).size())
hour_count.columns = ['hour_overall_offers']
hour_count['hour_key'] = hour_count.index
weekday_hour_count = pd.DataFrame(train_data.groupby(['weekday_key', 'hour_key']).size())
weekday_hour_count.columns = ['weekday_hour_overall_offers']

def add_count_per_day_factors(df):
    df = df.join(weekday_count, on='weekday_key', lsuffix='', rsuffix='_2').drop(['weekday_key_2'], axis=1)
    df = df.join(hour_count, on='hour_key', lsuffix='', rsuffix='_2').drop(['hour_key_2'], axis=1)
    df = df.join(weekday_hour_count, on=['weekday_key', 'hour_key'], lsuffix='', rsuffix='_2')
    return df