# Assignment 2: Sales Data Processing with PySpark DataFrames
# Original data: https://data.mendeley.com/datasets/kcgf45y24m/2
# Usage: py Assignment2.py Transactions.csv Products.csv 0.00506368
# The example currency conversion is a 2026-11-23 historical snapshot, not a live rate.

import sys
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, sum, count, desc, when, explode, split, trim, lower,
    regexp_replace, lit, concat
)

spark = (SparkSession.builder
         .appName('Sales Data Processing and Business Intelligence Using Spark DataFrames')
         .master('local[2]')
         .getOrCreate())
spark.sparkContext.setLogLevel('ERROR')

try:
    transactions_file = sys.argv[1] if len(sys.argv) > 1 else 'Transactions.csv'
    products_file = sys.argv[2] if len(sys.argv) > 2 else 'Products.csv'
    transactions = spark.read.csv(transactions_file, header=True, inferSchema=True)
    products = spark.read.csv(products_file, header=True, inferSchema=True)

    for old, new in [('ReceiptNumber', 'Receipt Number'),
                     ('NetSales', 'Net Sales'), ('TotalAmount', 'Total Amount')]:
        if old in transactions.columns and new not in transactions.columns:
            transactions = transactions.withColumnRenamed(old, new)
    if 'ProductName' in products.columns and 'Product Name' not in products.columns:
        products = products.withColumnRenamed('ProductName', 'Product Name')

    print('\n1. Original transaction schema')
    transactions.printSchema()
    print('\nOriginal product schema')
    products.printSchema()
    print('Transactions:', transactions.count())
    print('Product catalogue rows:', products.count())
    transactions.show(5, truncate=False)

    required_transactions = ['Receipt Number', 'Outlet', 'Date', 'Items',
                             'Net Sales', 'Total Amount', 'TotalItem']
    required_products = ['Product Name', 'Category', 'Price']
    missing_tx = [name for name in required_transactions if name not in transactions.columns]
    missing_products = [name for name in required_products if name not in products.columns]
    if missing_tx or missing_products:
        raise ValueError('Unexpected source CSV headings. Transactions missing: %s; '
                         'Products missing: %s' % (missing_tx, missing_products))

    print('\n2. Check missing fields')
    for name in required_transactions:
        print(name, 'null:', transactions.filter(col(name).isNull()).count())

    transactions = transactions.withColumn(
        'NetSalesIDR', regexp_replace(col('Net Sales').cast('string'), ',', '').cast('double')
    ).withColumn(
        'ReceiptAmountIDR', regexp_replace(col('Total Amount').cast('string'), ',', '').cast('double')
    )
    print('Invalid/missing Net Sales:',
          transactions.filter(col('NetSalesIDR').isNull()).count())
    valid = transactions.filter(col('NetSalesIDR').isNotNull() & (col('NetSalesIDR') >= 0))
    print('Valid receipts:', valid.count())

    print('\n3. Revenue by outlet: actual recorded net sales (IDR)')
    outlet_revenue = valid.groupBy('Outlet').agg(
        count('Receipt Number').alias('Receipts'),
        sum('NetSalesIDR').alias('NetSalesIDR')
    ).orderBy(desc('NetSalesIDR'))
    outlet_revenue.show(truncate=False)

    print('\n4. Highest-value receipts (IDR)')
    valid.select('Receipt Number', 'Outlet', 'Date', 'NetSalesIDR') \
        .orderBy(desc('NetSalesIDR')).show(10, truncate=False)

    print('\n5. Item names and catalogue categories')
    item_rows = valid.withColumn(
        'ItemName', explode(split(
            regexp_replace(col('Items').cast('string'), r"[\[\]']", ''), ','
        ))
    ).withColumn('ItemName', trim(col('ItemName')))
    item_rows = item_rows.filter(col('ItemName') != '')
    item_rows = item_rows.withColumn('JoinName', lower(trim(col('ItemName'))))

    if 'Variant' in products.columns:
        catalogue_name = when(
            col('Variant').isNull() |
            (trim(col('Variant')) == '') |
            (trim(col('Variant')) == '-'), col('Product Name')
        ).otherwise(concat(col('Product Name'), lit(' ('), col('Variant'), lit(')')))
    else:
        catalogue_name = col('Product Name')
    catalogue = products.withColumn(
        'JoinName', lower(trim(catalogue_name))
    ).withColumn(
        'CataloguePriceIDR',
        regexp_replace(col('Price').cast('string'), ',', '').cast('double')
    ).select('JoinName', 'Product Name', 'Category', 'CataloguePriceIDR')

    ambiguous = catalogue.groupBy('JoinName').count().filter(col('count') > 1).count()
    if ambiguous:
        raise ValueError('Catalogue contains %d duplicate matching names' % ambiguous)

    merged = item_rows.join(catalogue, 'JoinName', 'left')
    print('Representative view (item mentions, not verified quantities):')
    merged.select('Receipt Number', 'Date', 'Outlet', 'ItemName',
                  'Category', 'CataloguePriceIDR').show(10, truncate=False)
    mentions = merged.count()
    matches = merged.filter(col('Category').isNotNull()).count()
    print('Basket item mentions:', mentions)
    print('Matched catalogue mentions:', matches)
    print('Unmatched catalogue mentions:', mentions - matches)

    print('\n6. Best-selling items by recorded mentions (not actual quantity)')
    merged.groupBy('ItemName').agg(count('ItemName').alias('Mentions')) \
        .orderBy(desc('Mentions')).show(10, truncate=False)

    print('\n7. Category-wise catalogue-price estimate (not actual revenue)')
    merged.filter(col('Category').isNotNull()).groupBy('Category').agg(
        count('ItemName').alias('ItemMentions'),
        sum('CataloguePriceIDR').alias('EstimatedCatalogueValueIDR')
    ).orderBy(desc('EstimatedCatalogueValueIDR')).show(truncate=False)

    print('\n8. Revenue per receipt from recorded net sales')
    valid.select('Receipt Number', 'Outlet', 'NetSalesIDR').show(10, truncate=False)

    print("\n9. High-value transactions above INR 10,000")
    high_transactions = valid.filter(
        col("NetSalesIDR") > 10000
    ).orderBy(desc("NetSalesIDR"))

    print("\nTransactions above INR 10,000:")

    high_transactions.select(
        "Receipt Number", "Outlet", "Date", "NetSalesIDR"
    ).show(10, truncate=False)

    print(
        "Total transactions above INR 10,000:",
        high_transactions.count()
    )

    print("\nDaily outlet sales above INR 10,000")

    daily_sales = valid.groupBy(
        "Outlet", "Date"
    ).agg(
        sum("NetSalesIDR").alias("DailySalesIDR"),
        count("Receipt Number").alias("Transactions")
    )

    high_daily_sales = daily_sales.filter(
        col("DailySalesIDR") > 10000
    ).orderBy(desc("DailySalesIDR"))

    high_daily_sales.show(10, truncate=False)

    print(
        "Total outlet-days above INR 10,000:",
        high_daily_sales.count()
    )

    print('\n10. Product quantity and actual category revenue: matched single-item receipts')
    # When TotalItem == 1 AND only one parsed product is present, the recorded
    # receipt net sales can be attributed to that one product. This does NOT
    # establish the true unit quantity or line revenue for multi-item receipts.
    one_item_candidates = merged.filter(col('TotalItem') == 1)
    one_name_receipts = one_item_candidates.groupBy('Receipt Number').agg(
        count('ItemName').alias('ParsedItemCount')
    ).filter(col('ParsedItemCount') == 1).select('Receipt Number')
    single_item_sales = one_item_candidates.join(
        one_name_receipts, 'Receipt Number', 'inner'
    ).filter(col('Category').isNotNull())
    single_item_sales = single_item_sales.withColumn(
        'ObservedItemCount', lit(1)
    ).withColumn(
        'EffectiveNetPriceIDR', col('NetSalesIDR')
    ).withColumn(
        'SalesAmountIDR', col('ObservedItemCount') * col('EffectiveNetPriceIDR')
    )

    single_item_count = single_item_sales.count()
    print('Matched one-item receipts:', single_item_count)
    print('Coverage of original receipts: %.2f%%' %
          (100.0 * single_item_count / valid.count() if valid.count() else 0))
    print('One-item transaction examples:')
    single_item_sales.select('Receipt Number', 'Product Name', 'Category',
                             'Outlet', 'ObservedItemCount', 'EffectiveNetPriceIDR',
                             'Date', 'SalesAmountIDR').show(10, truncate=False)
    print('Actual recorded NET REVENUE by category on one-item receipts ONLY:')
    single_item_sales.groupBy('Category').agg(
        count('*').alias('OneItemReceipts'),
        sum('SalesAmountIDR').alias('SingleItemNetRevenueIDR')
    ).orderBy(desc('SingleItemNetRevenueIDR')).show(truncate=False)
    print('Top five products by matched one-item receipt frequency:')
    single_item_sales.groupBy('Product Name').agg(
        count('*').alias('OneItemReceipts')
    ).orderBy(desc('OneItemReceipts')).show(5, truncate=False)
    print('This does not provide all-basket product-level revenue or verified units sold.')

    print('\n11. Lazy evaluation and DataFrame execution plan')
    planned = valid.filter(col('NetSalesIDR') > 0).groupBy('Outlet').agg(
        sum('NetSalesIDR').alias('RevenueIDR')
    )
    print('Plan defined. explain(True) prints the plan without executing the query.')
    print('--- BEGIN explain(True) ---')
    planned.explain(True)
    print('--- END explain(True) ---')
    print('The following show() action executes the query:')
    planned.show(truncate=False)

    print('\n12. RDD and DataFrame revenue comparison')
    print('DataFrame outlet totals:')
    outlet_revenue.select('Outlet', 'NetSalesIDR').show(truncate=False)
    if sys.platform == 'win32' and sys.version_info >= (3, 12):
        print('Actual RDD calculation NOT RUN: Python 3.12 on Windows may not be '
              'compatible with this PySpark setup. Use Python 3.11 for a genuine RDD comparison.')
    else:
        try:
            rdd_totals = valid.select('Outlet', 'NetSalesIDR').rdd.map(
                lambda row: (row[0], row[1])
            ).reduceByKey(lambda a, b: a + b)
            print('Actual RDD outlet totals:', rdd_totals.collect())
        except Exception as error:
            print('Actual RDD calculation failed:', str(error))
            print('Do not report the DataFrame fallback as an RDD execution.')

    print('Review Jobs and SQL/DataFrame tabs at http://localhost:4040')
    input('Press ENTER after checking the Spark UI...')
finally:
    spark.stop()
