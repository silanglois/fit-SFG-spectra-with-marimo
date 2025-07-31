import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Fitting SFG Curves with Multiple Resonant Peaks""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    Using the iminuit package, select a window of SFG spectra and perform a non-linear fit of nonresonant and a variable number ($N^{res}$) of resonant peaks according to the equation

    \[
    \mathrm{SFG}(\omega) = \left| A_\text{NR} e^{i\phi} + \sum_{j=0}^{N^{res}}\frac{A^{res}_j}{\omega - \omega^{res}_j+i\Gamma_j}\right|^2
    \]

    where the parameters that we want to determine are the nonresonant amplitue ($A_\text{NR}$) and phase ($\phi$), as well as the amplitude, position, and width (damping constant) of each resonant peak ($A^{res}_j$, $\omega^{res}_j$, and $\Gamma_j$, respectively).

    This script was developed by Simon L.J. Langlois and the Cyran Lab. The fitting algorithm used here was originally posted on [Github](https://github.com/BoiseState-Chem/SFG_Fit.git).
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Prepare our script

    Import needed modules and build the required functions and objects
    """
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    import re
    import os
    import shutil
    from datetime import datetime
    from datetime import time as dtime

    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    import numpy as np
    import pandas as pd
    from iminuit import Minuit
    from typing import Callable
    return (
        Callable,
        GridSpec,
        Minuit,
        datetime,
        dtime,
        mo,
        np,
        os,
        pd,
        plt,
        re,
        shutil,
    )


@app.cell(hide_code=True)
def _(Callable, np):
    def curry(data: np.ndarray, func: Callable) -> Callable :
      def curriedfunc(*args):
        return func(data, *args)
      return curriedfunc
    return (curry,)


@app.cell(hide_code=True)
def _(np):
    def chi_non_resonant(nr: float, phase: float) -> np.complex128 :
      """
      Given the non-resonant parameters return a single complex-valued number
      for the non-resonant process
      """
      ChiNR = nr * np.exp(1j * phase)
      return ChiNR

    def chi_resonant(wavenumbers: np.ndarray[np.float64], amplitude: float, pos: float, width: float) -> np.ndarray[np.complex128]:
      """
      Given a range of wavenumbers and the parameters of a resonant peak return
      the complex values of the peak for each wavenumber
      """
      A = amplitude
      delta = wavenumbers - pos
      gamma = width / 2
      ChiR_im = -(A * gamma / (delta**2 + gamma**2))
      ChiR_re = A * delta / (delta**2 + gamma**2)
      ChiR = ChiR_re + (1j * ChiR_im)
      return ChiR
    return chi_non_resonant, chi_resonant


@app.cell(hide_code=True)
def _(chi_non_resonant, chi_resonant, np):
    def amplitude(wavenumbers: np.ndarray[np.float64], *args) -> np.ndarray[np.float64] :
      Chi = np.zeros(wavenumbers.shape, dtype=np.complex128)
      Chi = Chi + chi_non_resonant(args[0], args[1])
      nres = (len(args) - 2) // 3
      for i in range(nres):
        iarg = 3 * i + 2
        ChiR = chi_resonant(wavenumbers, args[iarg], args[iarg+1], args[iarg+2])
        Chi = Chi + ChiR
      return np.square(Chi.real) + np.square(Chi.imag)

    def chi(wavenumbers: np.ndarray[np.float64], *args) -> np.ndarray[np.float64]:
      Chi = np.zeros(wavenumbers.shape,dtype=np.complex128)
      Chi = Chi + chi_non_resonant(args[0], args[1])
      nres = (len(args)-2)//3
      for i in range(nres):
        iarg = 3*i+2
        ChiR = chi_resonant(wavenumbers, args[iarg], args[iarg+1], args[iarg+2])
        Chi = Chi + ChiR
      return Chi
    return amplitude, chi


@app.cell(hide_code=True)
def deffile(datetime, dtime, os, pd, re):
    class File:
        def __init__(self, directory: str, filename: str) -> None:
            """
        Initialize a File object.
        Args:
            directory: directory to the file
            filename: name of the file, including extension
        """
            # stores file information
            self.directory = directory
            self.filename = filename

            self.date: str | None = None

            self.label: str = filename # defaults to filename unless it gets modified

            self.sample: str | None = None
            self.acq_time: float | None = None
            self.polarization: str | None = None
            self.time: int | None = None
            self.region: float | None = None
            self.condition: str | None = None
            self.condition2: str | None = None
            self.concentration: str | None = None

            self.data: pd.DataFrame | None = None
            self.filtered_data: pd.DataFrame | None = None

            self.fitting_params: dict | None = None
            self.optimized_params: dict | None = None

        def __str__(self) -> str:
            return self.directory

        def extract_info(self, naming_pattern: str = (
                r"^([^_]+)"  # sample name (e.g. pfoa)
                r"(?:_([0-9]*\.?[0-9]+[a-zA-Z]+))?"  # optional valued unit: float + unit
                r"(?:_([^_]+))?"  # optional condition
                r"(?:_([^_]+))?"  # optional 2nd condition
                r"_([sp]{3})"  # polarization
                r"_([0-9]*\.?[0-9]+)um"  # region in um
                r"_([0-9]*\.?[0-9]+)s"  # acquisition time in s
                r"_([0-2][0-9][0-5][0-9])"  # time in hhmm (in 24hrs/military time)
                r"(?:_[\w]+)?"  # to handle any possibly following words
                r"\.csv$"
        )) -> None:
            """
        Extract information from the name of the file.
        Args:
            naming_pattern: regular expression pattern to extract information from
        """
            naming_match = re.match(naming_pattern, self.filename)
            if naming_match:
                (self.sample, self.concentration, self.condition, self.condition2, self.polarization,
                 region_str, acq_time_str, time) = naming_match.groups()
                self.region = float(region_str)
                self.acq_time = float(acq_time_str)
                self.time = datetime.strptime(time, "%H%M").time() #int(time)
            else:
                print(f'Warning: The name of {self.filename} does not match the naming pattern indicated.')

        def extract_data(self, delimiter: str = ',') -> None:
            """
        Extract data from the file and store it in a pandas.DataFrame.
        Args:
            delimiter: delimiter for the data in the file
        """
            self.data = pd.read_csv(self.directory, delimiter=delimiter, index_col=False)

        def extract_date(self) -> None:
            """
        Extract date from the file directory if it can be retreived.
        Assuming the directory include date information in the form of /yyyy_mm_dd/
        """
            date_pattern = re.compile(r'^\d{4}_\d{2}_\d{2}$')
            # Start at the directory containing the file
            current_dir = os.path.dirname(os.path.abspath(self.directory))
            while True:
                folder_name = os.path.basename(current_dir)
                if date_pattern.match(folder_name):
                    self.date = datetime.strptime(folder_name, "%Y_%m_%d").date()
                # Move one level up
                parent_dir = os.path.dirname(current_dir)
                if parent_dir == current_dir:  # reached root
                    break
                current_dir = parent_dir

        def build_label(self, *args: str) -> None:
            parts = []
            for arg in args:
                value = getattr(self, arg, None)
                if value is not None:
                    if arg == "time" and isinstance(value, dtime):
                        parts.append(value.strftime("%H:%M"))
                    else:
                        parts.append(str(value))
            if None not in parts:
                self.label = ' '.join(parts)


    return (File,)


@app.cell(hide_code=True)
def _(datetime, os, shutil):
    def backup_script(export_dir: str) -> str:
        try:
            script_path = __file__
        except AttributeError:
            raise RuntimeError("__file__ not available.")

        script_path = os.path.abspath(script_path)
        directory, filename = os.path.split(script_path)
        name, ext = os.path.splitext(filename)

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        new_filename = f"{name}_backup_{timestamp}{ext}"
        new_filepath = os.path.join(export_dir, new_filename)

        shutil.copy2(script_path, new_filepath)
        return f"Backup created at: {new_filepath}"
    return (backup_script,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Load Data""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
    /// attention | Warning

    The datafiles you import have to be of type .csv and have columns labeled "Wavenumber" (corresponding to the mid-IR wavenumber) and "Intensity" (containing the corresponding the vSFG intensity). There can be other columns, but they will be ignored.

    If your files are set up differently, you will need to modify them or modify this script.
    ///

    /// details | **File set-up tips**
        type: info

    If your files follow a specific naming pattern, this code is intended to extract information from the naming pattern. The default file naming system is **sample_conc_condition_condition2_polarization_region_acqlength_time.csv** with:

     - **sample** being any combination of letters/number (no underscores), intended to describe your sample species (e.g. *pfoa*).

     - **conc** (*optional*) being a combination of a number followed by letters, intended to decribe a concentration (e.g *1.5mM*).

     - **condition** (*optional*) and **condition2** (*optional*) being any combination of letters/number (no underscores), intended to describe your experimental conditions (e.g. *cold*, *h2o*).

     - **polarization** being any 3 letter combination of the letters *p* and *s*, intended to indicate light polarization (e.g. *ssp*).

     - **region** being the combination of a number and *um*, intended to indicate the infrared center wavelength (e.g. 6um).

     - **acqlength** being the combination of a number and *s*, intended to indicate the duration of acquisition in seconds (e.g. *600s*).

     - **time** being any combination of four digits, intented to indicate the time (in 24hr time) at which the acquisition was started (e.g. *1526*, for a spectrum acquired at 3:26 pm).

     - There can be additional words/numers between **time** and **.csv**, those will be ignored by default.

    This naming pattern can be modified using regular expression to suit your naming system, see the [File object](http://127.0.0.1:2718/#scrollTo=deffile) a few cells above. This script should still work if your files do not match the pattern indicated, but some features will not be available.

    **Note:** The date at which the spectra was acquired is extracted with the requirement that the path to the file is as follow: "some path/yyyy_mm_dd/some path/file.csv"

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    file_browser = mo.ui.file_browser(
        initial_path="C:/Users/simonlanglois/Desktop/simon/Data/",
        multiple=True,
        filetypes=['.csv']
    ).form()
    file_browser
    return (file_browser,)


@app.cell(hide_code=True)
def _(mo):
    _options = ['sample', 'concentration', 'condition', 'condition2', 'polarization', 'region', 'acq_time', 'date', 'time']
    _value = ['sample', 'concentration', 'condition', 'condition2', 'date', 'time']
    file_label = mo.ui.multiselect(options=_options, value=_value, label='File label include:')
    return (file_label,)


@app.cell(hide_code=True)
def _(File, file_browser, file_label, mo):
    datafiles = []

    _paths = [str(file.path) for file in file_browser.value]
    _names = [file.name for file in file_browser.value]

    for path, name in zip(_paths, _names):
        _file = File(directory=path, filename=name)
        _file.extract_data()
        _file.extract_info()
        _file.extract_date()
        _file.build_label(*file_label.value)
        datafiles.append(_file)

    mo.md(f'{file_label}\n\n For example, the label for **{datafiles[0].filename}** would be **{datafiles[0].label}**')
    return (datafiles,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Plot data""")
    return


@app.cell(hide_code=True)
def _(mo, plt):
    cmap1 = mo.ui.dropdown(options=plt.colormaps(), value='viridis', searchable=True, label='Colormap:')
    dpi1 = mo.ui.number(label='figure dpi:', value=125, step=25, start=50, stop=600)
    mo.md(f'For each plot, you can pick the colormap used to color your different lines.\n\n Try it! &#x2192; {cmap1}\n\n **Note:** The colormaps available come from the [matplotlib library](https://matplotlib.org/stable/users/explain/colors/colormaps.html).')
    return cmap1, dpi1


@app.cell(hide_code=True)
def _(cmap1, datafiles, dpi1, mo, np, plt):
    _fig, _ax = plt.subplots(figsize=(8.4, 4.5), dpi=dpi1.value)
    _colors = [plt.get_cmap(cmap1.value)(i) for i in np.linspace(0.05, 0.95, len(datafiles))]
    for _file, _color in zip(datafiles, _colors):
        _file.data.plot('Wavenumber', 'Intensity', label=_file.label, ax=_ax, color=_color)
    _ax.legend(fontsize=8)
    mo.vstack([mo.mpl.interactive(_fig), dpi1])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Filter your data""")
    return


@app.cell(hide_code=True)
def _(datafiles, mo, np):
    _wn = np.array(datafiles[0].data['Wavenumber'])

    min_wn =  mo.ui.number(label=r'$\tilde{\nu}_{min}$', value=np.round(np.min(_wn), 0))
    max_wn = mo.ui.number(label=r'$\tilde{\nu}_{max}$', value=np.round(np.max(_wn), 0))

    mo.vstack([
        mo.md('### Select range:'),
        min_wn,
        max_wn,
              ])
    return max_wn, min_wn


@app.cell(hide_code=True)
def _(datafiles, max_wn, min_wn):
    # filter the data using the selected range
    WMin = min_wn.value
    WMax = max_wn.value
    for _file in datafiles:
        _file.filtered_data = _file.data.query(f'Wavenumber > {WMin} and Wavenumber < {WMax}')
    return WMax, WMin


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### Display filtered data""")
    return


@app.cell(hide_code=True)
def _(mo, plt):
    cmap2 = mo.ui.dropdown(options=plt.colormaps(), value='viridis', searchable=True, label='[Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html):')
    dpi2 = mo.ui.number(label='figure dpi:', value=125, step=25, start=50, stop=600)
    cmap2
    return cmap2, dpi2


@app.cell(hide_code=True)
def _(WMax, WMin, cmap2, datafiles, dpi2, mo, np, plt):
    _ = WMin + WMax #this line serves no purpose other than having this cell be re-run if selected range is modified

    _fig, _ax = plt.subplots(figsize=(8.4, 4.5), dpi=dpi2.value)
    _colors = [plt.get_cmap(cmap2.value)(i) for i in np.linspace(0.05, 0.95, len(datafiles))]
    for _file, _color in zip(datafiles, _colors):
        _file.filtered_data.plot('Wavenumber', 'Intensity', label=_file.label, ax=_ax, color=_color)
    _ax.legend(fontsize=8)
    mo.vstack([mo.mpl.interactive(_fig), dpi2])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Perform the fit""")
    return


@app.cell(hide_code=True)
def _(mo):
    num_res = mo.ui.number(label=r'$N^{\text{res}}$ = ', start=1, value=1, step=1)
    return (num_res,)


@app.cell(hide_code=True)
def _(WMax, WMin, mo, np, num_res):
    values ={
        'nr_amplitude': mo.ui.number(label=r'$A_{NR}=$', value=None),
        'nr_phase': mo.ui.number(label=r'$\phi_{NR}=$', value=0, start=-np.pi, stop=np.pi),
    }

    fixs ={
        'nr_amplitude': mo.ui.checkbox(label='fixed', value=True),
        'nr_phase': mo.ui.checkbox(label='fixed', value=True),
    }

    mins ={
        'nr_amplitude': mo.ui.number(label='Min:', value=0),
        'nr_phase': mo.ui.number(label='Min:', value=-np.pi),
    }

    maxs ={
        'nr_amplitude': mo.ui.number(label='Max:', value=None),
        'nr_phase': mo.ui.number(label='Max:', value=np.pi),
    }

    for _n in range(num_res.value):
        _n += 1
        values[f'r{_n}_amplitude'] = mo.ui.number(label=rf'$A_{_n}=$', value=None)
        values[f'r{_n}_pos'] = mo.ui.number(label=rf'$\omega_{_n}=$', start=WMin, stop=WMax)
        values[f'r{_n}_width'] = mo.ui.number(label=rf'$\Gamma_{_n}=$', value=None)

        fixs[f'r{_n}_amplitude'] = mo.ui.checkbox(label='fixed')
        fixs[f'r{_n}_pos'] = mo.ui.checkbox(label='fixed')
        fixs[f'r{_n}_width'] = mo.ui.checkbox(label='fixed')

        mins[f'r{_n}_amplitude'] = mo.ui.number(label='Min:', value=None)
        mins[f'r{_n}_pos'] = mo.ui.number(label='Min:', value=WMin)
        mins[f'r{_n}_width'] = mo.ui.number(label='Min:', value=0)

        maxs[f'r{_n}_amplitude'] = mo.ui.number(label='Max:', value=None)
        maxs[f'r{_n}_pos'] = mo.ui.number(label='Max:', value=WMax)
        maxs[f'r{_n}_width'] = mo.ui.number(label='Max:', value=None)


    parameters = mo.ui.dictionary({
        'values': mo.ui.dictionary(values),
        'fixs': mo.ui.dictionary(fixs),
        'mins': mo.ui.dictionary(mins),
        'maxs': mo.ui.dictionary(maxs),
    })
    return (parameters,)


@app.cell(hide_code=True)
def _(mo, num_res, parameters):
    # parameters = parameters_form.value

    _keys = ['nr_amplitude', 'nr_phase'] + [
        _key1
        for _n in range(num_res.value)
        for _key1 in (f'r{_n+1}_amplitude', f'r{_n+1}_pos', f'r{_n+1}_width')
    ]

    _params = [f"{parameters['values'][_key]} {parameters['fixs'][_key]} {parameters['mins'][_key]} {parameters['maxs'][_key]}" for _key in _keys]

    mo.md(
        f"### Initial fitting parameters \n {num_res} \n\n **Non resonant**\n\n"
        + '\n\n'.join(
            ('\n\n'+rf'**Resonant feature \#{_k[1]}**'+'\n\n' if (_k.startswith('r') and _k.endswith('_amplitude')) else '') + _p
        for _p, _k in zip(_params, _keys)
        )
    )
    return


@app.cell(hide_code=True)
def _(mo, parameters):
    mo.md(
        rf"""
    *For troubleshooting* - **values** = {parameters.value['values']}

    <!-- **fixed** {parameters.value['fixs']}

    **mins** {parameters.value['mins']}

    **maxs** {parameters.value['maxs']} -->
    """
    )
    return


@app.cell(hide_code=True)
def _(datafiles, parameters):
    param_fixed: list = [_p for _p in parameters.value.get('fixs').keys() if parameters.value.get('fixs')[_p] == True]

    param_lims: dict = {_key: (parameters.value.get('mins')[_key], parameters.value.get('maxs')[_key]) for _key in parameters.value.get('mins')}

    param_values: dict = parameters.value.get('values')

    for _param_val in param_values.values():
        if _param_val != None:
            callout_text = '''
            /// admonition | Note
            If initial parameters are not all provided, the cell below will raise an error (but everything is looking good :sunflower:)
            ///
            '''
        else:
            callout_text = '''
            /// attention | Warning!
            It appears that one of the parameters provided is invalid
            ///
            '''
            break

    for _file in datafiles:
        _file.fitting_params = param_values
    return callout_text, param_fixed, param_lims


@app.cell(hide_code=True)
def _(callout_text, mo):
    run_button = mo.ui.run_button(label='**Click to run**')
    mo.hstack([
        mo.callout(mo.center(run_button), kind='info'),
        mo.md(callout_text)
    ], justify='start', gap=1, align='center')
    return (run_button,)


@app.cell(hide_code=True)
def _(
    Minuit,
    amplitude,
    chi,
    curry,
    datafiles,
    mo,
    np,
    param_fixed: list,
    param_lims: dict,
    pd,
    run_button,
):
    def costfunction_of_sfg(intensity: np.ndarray[np.float64], *args) -> np.float64 :
      return np.sum((intensity - calcamplitude(*args)) ** 2)

    mo.stop(not run_button.value, mo.md("Click 👆 to fit your data"))

    fit_results = []

    for _file in datafiles:
        wavenumbers = np.array(_file.filtered_data['Wavenumber'])
        intensity = np.array(_file.filtered_data['Intensity'])
        calcamplitude = curry(wavenumbers, amplitude)
        calc_chi = curry(wavenumbers, chi)
        costfunction = curry(intensity, costfunction_of_sfg)
        costfunction.errordef = Minuit.LEAST_SQUARES

        # Set up the fit
        m = Minuit(costfunction, name=_file.fitting_params.keys(), *_file.fitting_params.values())

        for _p, _lims in param_lims.items():
            m.limits[_p] = _lims

        for _p in param_fixed:
            m.fixed[_p] = True

        # perform the fit
        m.migrad()

        # extract fit results
        _df = pd.DataFrame({
            "Parameter": m.parameters,
            "Value": [m.values[p] for p in m.parameters],
            "Error": [m.errors[p] for p in m.parameters],
            "Fixed": [m.fixed[p] for p in m.parameters],
        })
        fit_results.append(mo.vstack([
            mo.md(f'{_file.label}'),
            m.fmin,
            m.covariance,
            mo.as_html(_df)
        ]))

        _file.optimized_params = dict(zip(_file.fitting_params.keys(), [p.value for p in m.params]))

    mo.md("Click 👆 to fit your data")
    return calc_chi, calcamplitude, fit_results


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Fit results""")
    return


@app.cell(hide_code=True)
def _(fit_results, mo):
    mo.carousel(fit_results)
    # Carousel is nice but I don't know how to show slide bar in case tables are larger than screen. If that is your case, that sucks :)
    return


@app.cell(hide_code=True)
def _(datafiles, fit_results, mo, pd):
    _ = fit_results #for cell update only
    _summary_table = pd.DataFrame([
        {'file': _file.label, **_file.optimized_params}
        for _file in datafiles
    ]).set_index('file')
    mo.vstack([mo.md('### Summary'), _summary_table])
    return


@app.cell(hide_code=True)
def _(datafiles, mo, plt):
    _options = {_f.label: _f for _f in datafiles}
    files_chosen = mo.ui.multiselect(options=_options, value=_options, label='Select which file(s) to display:')
    display_residuals1 = mo.ui.checkbox(label='Display residuals', value=True)
    display_im1 = mo.ui.checkbox(label=r'Display Im$(\chi^{(2)}_\text{eff})$ (dashed lines)', value=False)
    display_re1 = mo.ui.checkbox(label=r'Display Re$(\chi^{(2)}_\text{eff})$ (dotted lines)', value=False)
    cmap3 = mo.ui.dropdown(options=plt.colormaps(), value='viridis', searchable=True, label='[Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html):')

    dpi3 = mo.ui.number(label='figure dpi:', value=125, step=25, start=50, stop=600)

    mo.vstack([
        mo.md('## Plot fit results'),
        files_chosen,
        cmap3,
        display_residuals1,
        display_im1,
        display_re1,
    ])
    return (
        cmap3,
        display_im1,
        display_re1,
        display_residuals1,
        dpi3,
        files_chosen,
    )


@app.cell(hide_code=True)
def _(
    GridSpec,
    calc_chi,
    calcamplitude,
    cmap3,
    display_im1,
    display_re1,
    display_residuals1,
    dpi3,
    files_chosen,
    mo,
    np,
    plt,
):
    _fig = plt.figure(figsize=(8.4, 4.5), dpi=dpi3.value)

    if display_residuals1.value:
        _gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.05)
        _ax = _fig.add_subplot(_gs[0])
        _res_ax = _fig.add_subplot(_gs[1], sharex=_ax)
    else:
        _ax = _fig.add_subplot()

    _colors = [plt.get_cmap(cmap3.value)(i) for i in np.linspace(0.05, 0.95, len(files_chosen.value))]

    for _color, _file in zip(_colors, files_chosen.value):
        _df = _file.filtered_data
        _x = _df['Wavenumber']
        _y = _df['Intensity']
        _y_fit = calcamplitude(*_file.optimized_params.values())
        _chi = calc_chi(*_file.optimized_params.values())

        _ax.plot(_x, _y, alpha=0.3, marker='.', markersize=5, color=_color)
        _ax.plot(_x, _y_fit, linewidth=2, label=_file.label, color=_color)

        if display_im1.value:
            _ax.plot(_x, _chi.imag, ls='--', color=_color)
        if display_re1.value:
            _ax.plot(_x, _chi.real, ls=':', color=_color)
        if display_residuals1.value:
            _res_ax.plot(_x, _y - _y_fit, marker='.', linewidth=1, markersize=2, alpha=0.8, color=_color)

    _ax.set_ylabel("Intensity (a.u.)")
    _ax.legend(fontsize=8)

    if display_residuals1.value:
        plt.setp(_ax.get_xticklabels(), visible=False)
        _res_ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        _res_ax.set_ylabel("Residuals")
        _res_ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    else:
        _ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")

    mo.vstack([mo.mpl.interactive(_fig), dpi3])
    return


@app.cell(hide_code=True)
def _(datafiles, mo, plt):
    _options = {_f.label: _f for _f in datafiles}
    file_chosen = mo.ui.dropdown(options=_options, value=next(iter(_options)), label='Choose a file:')
    display_residuals2 = mo.ui.checkbox(label='Display residuals', value=False)
    display_im2 = mo.ui.checkbox(label=r'Display Im$(\chi^{(2)}_\text{eff})$', value=True)
    display_re2 = mo.ui.checkbox(label=r'Display Re$(\chi^{(2)}_\text{eff})$', value=True)
    display_phase = mo.ui.checkbox(label=r'Display $\chi^{(2)}$ phase', value=False)
    display_resonants = mo.ui.checkbox(label='Display individual resonant features', value=False)
    cmap4 = mo.ui.dropdown(options=plt.colormaps(), value='viridis', searchable=True, label='[Colormap](https://matplotlib.org/stable/users/explain/colors/colormaps.html):')

    dpi4 = mo.ui.number(label='figure dpi:', value=125, step=25, start=50, stop=600)

    mo.vstack([
        mo.md('## Focus on a single datafile'),
        file_chosen,
        display_residuals2,
        display_im2,
        display_re2,
        display_phase,
        mo.hstack([display_resonants, cmap4], justify='start', gap=4)
    ])
    return (
        cmap4,
        display_im2,
        display_phase,
        display_re2,
        display_residuals2,
        display_resonants,
        dpi4,
        file_chosen,
    )


@app.cell(hide_code=True)
def _(
    GridSpec,
    calc_chi,
    calcamplitude,
    chi_non_resonant,
    chi_resonant,
    cmap4,
    display_im2,
    display_phase,
    display_re2,
    display_residuals2,
    display_resonants,
    dpi4,
    file_chosen,
    mo,
    np,
    num_res,
    plt,
):
    _fig = plt.figure(figsize=(8.4, 4.5), dpi=dpi4.value)

    if display_residuals2.value:
        _gs = GridSpec(nrows=2, ncols=1, height_ratios=[3, 1], hspace=0.05)
        _ax = _fig.add_subplot(_gs[0])
        _res_ax = _fig.add_subplot(_gs[1], sharex=_ax)
    else:
        _ax = _fig.add_subplot()

    _file = file_chosen.value
    _df = _file.filtered_data
    _x = _df['Wavenumber']
    _y = _df['Intensity']
    _y_fit = calcamplitude(*_file.optimized_params.values())
    _chi = calc_chi(*_file.optimized_params.values())
    _chi_nr = chi_non_resonant(
        nr=_file.optimized_params['nr_amplitude'],
        phase=_file.optimized_params['nr_phase']
    )
    #data
    _ax.plot(_x, _y, alpha=0.6, marker='.', markersize=8, color='#ff8400', label='data')

    #fit
    _ax.plot(_x, _y_fit, linewidth=2, alpha=0.8, color='#010342', label='fit')

    #residuals
    if display_residuals2.value:
        _res_ax.plot(_x, _y - _y_fit, marker='.', linewidth=1, markersize=2, color='#010342')

    #imaginary
    if display_im2.value:
        _ax.plot(_x, _chi.imag, ls='--', color='blue', label=r'Im$\chi^{(2)}_{\text{eff}}$')

    #real
    if display_re2.value:
        _ax.plot(_x, _chi.real, ls='--', color='red', label=r'Re$\chi^{(2)}_{\text{eff}}$')

    #resonants
    if display_resonants.value:
        _colors = [plt.get_cmap(cmap4.value)(i) for i in np.linspace(0.05, 0.95, num_res.value)]
        for _n, _color in zip(range(num_res.value), _colors):
            _n += 1
            _chi_rN = chi_resonant(
                wavenumbers=np.array(_df['Wavenumber']),
                amplitude=_file.optimized_params[f'r{_n}_amplitude'],
                pos=_file.optimized_params[f'r{_n}_pos'],
                width=_file.optimized_params[f'r{_n}_width']
            ) + _chi_nr
            _ax.plot(_x, np.square(_chi_rN.real)+np.square(_chi_rN.imag), label=rf'$|\chi^{{(2)}}_{{\text{{R{_n}}}}}|^2$', color=_color)

    if display_phase.value:
        _ax2 = _ax.twinx()
        color2: str = '#0276b5'
        # color2: str = '#c70202'
        _ax2.set_ylabel(r'Phase (in $\pi$ radians)', color=color2)
        _ax2.plot(_x, np.angle(_chi) / np.pi, color=color2, label='Phase', alpha=0.8, zorder=1)
        _ax2.tick_params(axis='y', labelcolor=color2)

    _ax.set_title(_file.label)
    _ax.set_ylabel("Intensity (a.u.)")

    # Combine legends from both axes
    _handles1, _labels1 = _ax.get_legend_handles_labels()
    if display_phase.value:
        _handles2, _labels2 = _ax2.get_legend_handles_labels()
        _handles = _handles1 + _handles2
        _labels = _labels1 + _labels2
    else:
        _handles, _labels = _handles1, _labels1

    _legend = _fig.legend(_handles, _labels, fontsize=8, loc='upper left')
    _legend.set_zorder(10)


    if display_residuals2.value:
        plt.setp(_ax.get_xticklabels(), visible=False)
        _res_ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        _res_ax.set_ylabel("Residuals")
        _res_ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
    else:
        _ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")

    mo.vstack([mo.mpl.interactive(_fig), dpi4])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Saving the resulting fits""")
    return


@app.cell(hide_code=True)
def _():
    # _ = fit_results #for cell update only

    # # The intent of this cell is to save the fit results and sample info to a database which woud be helpful for further analysis. However this feature is not yet implemented, at this time only a table of present fits is shown. 
    # _df = pd.DataFrame([
    #     {
    #         'filename': _file.filename,
    #         'date': _file.date,
    #         'time': _file.time.strftime("%H:%M"),
    #         'sample': _file.sample,
    #         'concentration': _file.concentration,
    #         'condition': _file.condition,
    #         'condition2': _file.condition2,
    #         'polarization': _file.polarization,
    #         'region': _file.region,
    #         'acq_time': _file.acq_time,
    #         'num_resonance': num_res.value,
    #         **_file.optimized_params,
    #     }
    #     for _file in datafiles
    # ])#.set_index('filename')

    # df_to_save = mo.ui.data_editor(_df).form(bordered=False)

    # mo.vstack([
    #     mo.md("The intent of this cell is to save the fit results and sample info to a database which woud be helpful for further analysis. However this feature is not yet implemented, but here is a table."),
    #     df_to_save
    # ])
    return


@app.cell(hide_code=True)
def _(datafiles, mo):
    save_button = mo.ui.run_button(label='Save')

    _options = {_f.label: _f for _f in datafiles}
    to_save = mo.ui.multiselect(label= 'Select which fitted files you want to save', options=_options, value=_options)
    return save_button, to_save


@app.cell(hide_code=True)
def _(
    WMax,
    WMin,
    backup_script,
    mo,
    num_res,
    os,
    param_fixed: list,
    param_lims: dict,
    save_button,
    to_save,
):
    _accordion = {}

    for _file in to_save.value:
        _text = f"""
    This file contains information about the fit for '{_file.filename}'.
    ======================================================================
    Fitted from {WMin} to {WMax} wavenumber.

    Number of resonant features fitted: {num_res.value}

    Optimized parameters:
    {_file.optimized_params}

    Initial parameters provided:
    {_file.fitting_params}

    Fixed parameters:
    {param_fixed if len(param_fixed) > 0 else 'None'}

    Parameters limits:
    {param_lims}
        """
        _directory = os.path.dirname(os.path.dirname(_file.directory))  # Two times to go up two levels
        _export_directory = _directory + '\\Fitting\\'

        _accordion[f' **fit-{_file.filename.replace('.csv', '.txt')}**'] = mo.vstack([
            mo.md(f'**Directory:**'),
            mo.plain_text(f'{_export_directory}fit-{_file.filename.replace('.csv', '.txt')}'),
            mo.md(f'**File content:**'),
            mo.plain_text(_text)
        ])

        if save_button.value:
            os.makedirs(_export_directory, exist_ok=True)
            _backup = backup_script(_export_directory)
            _accordion[f' **fit-{_file.filename.replace('.csv', '.txt')}**'] = mo.md(_backup)
            with open(f'{_export_directory}fit-{_file.filename.replace('.csv', '.txt')}', 'w') as txt:
                txt.write(_text)

    mo.vstack([
        mo.md(f'''
        ### Preview the files to be saved\n\n
        {to_save}\n\n
        If satisfied, click the below button to save the following files. Also, a copy of this script will be saved next to each text file.
        '''),
        mo.left(mo.accordion(_accordion)),
        mo.md(r'''
        /// attention | Warning
        If a file with the same name and directory already exists, it will be replaced.
        ///
        '''),
        save_button
    ], align='start', gap=1)
    return


if __name__ == "__main__":
    app.run()
