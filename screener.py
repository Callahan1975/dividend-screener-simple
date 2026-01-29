<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Dividend Screener – Baseline</title>

    <!-- DataTables CSS -->
    <link rel="stylesheet" href="https://cdn.datatables.net/1.13.8/css/jquery.dataTables.min.css">

    <!-- jQuery -->
    <script src="https://code.jquery.com/jquery-3.7.1.min.js"></script>

    <!-- DataTables JS -->
    <script src="https://cdn.datatables.net/1.13.8/js/jquery.dataTables.min.js"></script>

    <!-- PapaParse (CSV parser) -->
    <script src="https://cdn.jsdelivr.net/npm/papaparse@5.4.1/papaparse.min.js"></script>

    <style>
        body {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            padding: 20px;
        }
        h1 {
            margin-bottom: 20px;
        }
    </style>
</head>
<body>

<h1>Dividend Screener – Baseline</h1>

<table id="screener" class="display" style="width:100%">
    <thead>
        <tr>
            <th>Ticker</th>
            <th>Name</th>
            <th>Country</th>
            <th>Sector</th>
            <th>Industry</th>
            <th>Price</th>
            <th>Dividend Yield (%)</th>
            <th>Payout Ratio (%)</th>
            <th>PE</th>
        </tr>
    </thead>
    <tbody></tbody>
</table>

<script>
document.addEventListener("DOMContentLoaded", function () {

    Papa.parse("screener_results.csv", {
        download: true,
        header: true,
        skipEmptyLines: true,
        complete: function (results) {

            const data = results.data.map(row => [
                row.Ticker,
                row.Name,
                row.Country,
                row.Sector,
                row.Industry,
                row.Price,
                row["Dividend Yield (%)"],
                row["Payout Ratio (%)"],
                row.PE
            ]);

            $('#screener').DataTable({
                data: data,
                pageLength: 25,
                order: [[0, "asc"]],
                columnDefs: [
                    { targets: [5,6,7,8], className: "dt-right" }
                ]
            });
        },
        error: function (err) {
            console.error("CSV load error:", err);
        }
    });

});
</script>

</body>
</html>
