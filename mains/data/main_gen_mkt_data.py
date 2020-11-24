from src.data import gen_mkt_data
import yaml
with open("configs/optimise_trading_rules.yml", "r") as ymlfile:
    cfg = yaml.load(ymlfile)
gen_mkt_data.run(includ_etfs=False).to_pickle(cfg['full_mkt_data'])
