"""Independent synthetic tests; no WRDS client, network, or licensed observations."""
import csv
import json
import tempfile
import unittest
from pathlib import Path
import pandas as pd
from wrds_extract import config, extract, probe

P=Path(__file__).parent
class FakeDB:
    def __init__(self,tables,rows=None,columns=None):
        self.tables=tables; self.rows=rows or []; self.columns=columns
        self.sql=[]
    def list_tables(self,library):
        assert library=='optionm'; return self.tables
    def describe_table(self,library,table):
        assert table in self.tables
        return pd.DataFrame({'name':self.columns or ['secid','date','days','delta','cp_flag','impl_volatility']})
    def raw_sql(self,sql):
        self.sql.append(sql); return pd.DataFrame(self.rows)

class VeraOfflineTests(unittest.TestCase):
    def setUp(self):
        self.c=config(P/'wrds_config.template.json')
        self.c.update(start_date='2020-01-01',end_date='2020-12-31',schema_confirmed=True,security_identity_confirmed=True)
    def test_select_manifest_and_scope(self):
        db=FakeDB(['vsurfd2020'],[{'secid':108105,'date':'2020-03-02','days':30,'delta':50,'cp_flag':'C','impl_volatility':0.25}])
        # A mocked raw_sql result uses the aliases returned by the actual SELECT.
        db.rows=[{'secid':108105,'date':'2020-03-02','days':30,'delta':50,'cp_flag':'C','iv':0.25}]
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/'new'; extract(db,self.c,out)
            manifest=json.loads((out/'manifest.json').read_text())
            self.assertEqual((manifest['files'][0]['queried_rows'],manifest['files'][0]['retained_rows']),(1,1))
            with (out/'optionmetrics_iv_2020.csv').open() as fh:
                self.assertEqual(list(csv.DictReader(fh))[0]['iv'],'0.25')
            self.assertEqual(len(db.sql),1)
            self.assertIn('WHERE secid IN (108105)',db.sql[0])
            self.assertIn("date >= '2020-01-01'",db.sql[0])
            self.assertIn("date <= '2020-12-31'",db.sql[0])
            self.assertIn("days = 30",db.sql[0]); self.assertIn("delta = 50",db.sql[0])
            self.assertIn("cp_flag = 'C'",db.sql[0])
            with self.assertRaises(FileExistsError): extract(db,self.c,out)
    def test_missing_year_and_column_fail_before_query(self):
        for db in (FakeDB([]),FakeDB(['vsurfd2020'],columns=['secid','date'])):
            with tempfile.TemporaryDirectory() as temp:
                with self.assertRaises(RuntimeError): extract(db,self.c,Path(temp)/'new')
                self.assertEqual(db.sql,[])
    def test_confirmation_string_rejected_before_metadata_or_query(self):
        c=dict(self.c,schema_confirmed='false',security_identity_confirmed='false')
        with tempfile.TemporaryDirectory() as temp:
            db=FakeDB(['vsurfd2020'],[{'secid':108105,'date':'2020-03-02','days':30,'delta':50,'cp_flag':'C','iv':0.25}])
            with self.assertRaises(ValueError): extract(db,c,Path(temp)/'new')
            self.assertEqual(db.sql,[])

if __name__=='__main__': unittest.main()
