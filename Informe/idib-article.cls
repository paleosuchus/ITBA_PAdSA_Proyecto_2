%% This is file `idib-article.cls'.
%%
%% ----------------------------------------------------------------
%% @file   idib-article.cls
%% @brief  Technical reports / article template for ITBA.
%% @author Ian Dib <idib@itba.edu.ar>
%% ----------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{idib-article}[2026/08/18 v5.0 ITBA report/article template]

% Options
\RequirePackage{kvoptions}
\SetupKeyvalOptions{family=idib,prefix=idib@}
\DeclareBoolOption[true]{cover}
\DeclareBoolOption[true]{color}
\DeclareBoolOption[true]{margin}
%% @brief Centre the cover and move the contents to a page of their own.
%% @note  The default cover packs title, contents, authors and date onto one page, which suits a short report.  A long one is better served by a cover that only announces the document and a contents page that can grow; `tocpage` asks for that arrangement.
\DeclareBoolOption[false]{tocpage}
% Presets are named combinations of the keys above and nothing else: each one only sets keys the document could have set by hand, so naming a situation never unlocks a layout the keys cannot reach.  A preset that touched anything not exposed as a key would make the two ways of asking disagree.
\DeclareVoidOption{journal}{%
  \idib@coverfalse\idib@colorfalse\PassOptionsToClass{twocolumn}{article}%
}
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessKeyvalOptions{idib}
\LoadClass[10pt]{article}

% The wide side margin needs a single column to live in, so two columns switch it off whatever margin says.  Geometry, the margin commands and the header offset all read this one boolean rather than asking the same question three times.
\newif\ifidib@sidemargin \idib@sidemargintrue
\if@twocolumn\idib@sidemarginfalse\fi
\ifidib@margin\else\idib@sidemarginfalse\fi

% Page geometry
\ifidib@sidemargin
  \RequirePackage[letterpaper,
                  left=0cm, right=0cm,
                  top=2.5cm, bottom=2.5cm,
                  marginparwidth=3.25cm, marginparsep=0.5cm]{geometry}
\else
  \if@twocolumn
    \RequirePackage[letterpaper,
                    left=1.8cm, right=5.2cm,
                    top=2.3cm, bottom=2cm,
                    columnsep=2em, headsep=0.8cm,
                    marginparwidth=0pt, marginparsep=0pt]{geometry}
  \else
    \RequirePackage[letterpaper,
                    left=2.5cm, right=2.5cm,
                    top=2.5cm, bottom=2.5cm,
                    marginparwidth=0pt, marginparsep=0pt]{geometry}
  \fi
\fi

% Shared infrastructure
\RequirePackage{idib-base}
\RequirePackage{titlesec}
\RequirePackage{titletoc}
\if@twocolumn
  \RequirePackage{cuted}                   % full-width blocks inside two columns
  \setlength{\stripsep}{\smallskipamount}  % air around the stripped abstract
\fi

% Monochrome: re-point the palette instead of branching at every use site.  The
% table tint and the logo follow the same switch, so nothing else has to ask
% which layout is running.
\newcommand{\idib@logo}{itba-logo-color}
\newcommand{\idib@mark}{itba-mark-color}
\ifidib@color\else
  \colorlet{navy}{black}
  \colorlet{gold}{black}
  \colorlet{cream}{white}
  \renewcommand{\idib@logo}{itba-logo}
  \renewcommand{\idib@mark}{itba-mark}
\fi

%% @brief Kind of document, printed in small caps above the title.  Leave the command out and nothing is printed.
%% @note  It answers "what am I holding" before the title answers "what is it about", which is why journals print it first and why a memoria and an informe built from the same class do not look alike.
\newcommand{\idib@articletype}{}
\newcommand{\articletype}[1]{\renewcommand{\idib@articletype}{#1}}

%% @brief Publication reference: where this document belongs, printed as a thin line above everything else.  Free text, so it carries a course and a term as readily as a journal, a volume and a page range.
\newcommand{\idib@pubref}{}
\newcommand{\pubref}[1]{\renewcommand{\idib@pubref}{#1}}

%% @brief Profesor o profesores de la cátedra, mostrados en la tapa debajo de la fecha.
\newcommand{\idib@profesores}{}
\newcommand{\profesores}[1]{\renewcommand{\idib@profesores}{#1}}

%% @brief The two lines above the title, shared by the cover and the inline block so both layouts open the same way.
\newcommand{\idib@frontsize}{\fontsize{7.5}{9}\selectfont}
\newcommand{\idib@typesize}{\small}
\colorlet{fronttype}{navy}
\newcommand{\idib@frontlines}{%
  \ifdefempty{\idib@pubref}{}{%
    {\sffamily\idib@frontsize\color{grey}\idib@pubref\par}%
    \smallskip
  }%
  \ifdefempty{\idib@articletype}{}{%
    {\sffamily\idib@typesize\bfseries\color{fronttype}%
     \MakeUppercase{\idib@articletype}\par}%
    \smallskip
  }%
}

% Metadata beyond the kernel's \title, \author and \date Authors.  One \author per person; the optional fields degrade to nothing when left out.  Each call appends an entry to \idib@authors, and \idib@entry is redefined per context, so the same list renders as a name line, as cover rows or as the detail block under an inline title.
%% @brief Add an author.
%% @param name    the author's name.
%% @param email   optional e-mail address.
%% @param id      optional student number (legajo).
%% @param note    optional extra line of information.
\newcommand{\idib@authors}{}
\newcommand{\idib@entry}[4]{}
\RenewDocumentCommand{\author}{m O{} O{} O{}}{%
  \appto\idib@authors{\idib@entry{#1}{#2}{#3}{#4}}%
}
%% @brief The author names on one line, comma separated.
\newcommand{\idib@authorline}{%
  \begingroup
    \def\idib@sep{}%
    \renewcommand{\idib@entry}[4]{\idib@sep ##1\def\idib@sep{, }}%
    \idib@authors
  \endgroup
}
% The marked name line and the detail block below it walk the same list and step the same counter, so every superscript finds its line and an author without details carries no marker at all.
\newcounter{idib@au}
%% @brief The author names with a superscript marker on each author that has details to show, for the inline title block.
\newcommand{\idib@authorline@marked}{%
  \begingroup
    \setcounter{idib@au}{0}\def\idib@sep{}%
    \renewcommand{\idib@entry}[4]{%
      \idib@sep ##1%
      \ifblank{##2##3##4}{}{\stepcounter{idib@au}\textsuperscript{\theidib@au}}%
      \def\idib@sep{, }%
    }%
    \idib@authors
  \endgroup
}
%% @brief The detail block: one small line per author with details, keyed to the marker rather than repeating the name.
\newcommand{\idib@authordetails}{%
  \begingroup
    \setcounter{idib@au}{0}%
    \renewcommand{\idib@entry}[4]{%
      \ifblank{##2##3##4}{}{%
        \stepcounter{idib@au}\textsuperscript{\theidib@au}\,%
        \def\idib@sep{}%
        \ifblank{##2}{}{##2\def\idib@sep{\enspace$\cdot$\enspace}}%
        \ifblank{##3}{}{\idib@sep Legajo~##3\def\idib@sep{\enspace$\cdot$\enspace}}%
        \ifblank{##4}{}{\idib@sep ##4}%
        \par
      }%
    }%
    \idib@authors
  \endgroup
}

% Header content, defined once and reused by the running and first-page styles.
% Either field may be left empty, and the separator only appears between two
% present ones: a document that carries its institution on the cover has no
% reason to repeat it on every page, and an empty field must not leave a stray
% dot behind.
\newcommand{\idib@head@left}{%
  \sffamily\footnotesize
  \ifodd\c@page
    \ifdefempty{\@institution}{%
      \nouppercase{\@subject}%
    }{%
      \textbf{\@institution}%
      \ifdefempty{\@subject}{}{\enspace$\cdot$\enspace\nouppercase{\@subject}}%
    }%
  \else
    \idib@authorline
  \fi
}
\newcommand{\idib@head@right}{\sffamily\footnotesize\textbf{\thepage}}

%% @brief Small grey label heading a metadata field.  The cover and the inline title block both build their front matter out of these label/value pairs, which is what makes the two layouts read as one family.
\newcommand{\idib@metalabel}[1]{%
  {\sffamily\fontsize{6.5}{8}\selectfont\bfseries\color{grey}\MakeUppercase{#1}\par}%
  \smallskip
}

% Table of contents
%% @brief Contents formatting.  The entry runs from the label to the title; what follows the title depends on where the contents sit.
%% @note  On a cover the contents are a short list under the title and the folio reads better right after it.  On a page of their own the line is far wider than the entry, and a folio left hanging in the middle of it loses the reader; there the number goes to the right margin with a leader.
% True only when the contents sit on the cover, under the title.
\newif\ifidib@covertoc
\ifidib@cover\ifidib@tocpage\else\idib@covertoctrue\fi\fi

\newcommand{\idib@tocformat}[2]{%
  % #1 sectional unit, #2 indent
  % The folio hangs at the right margin with a leader whenever the contents
  % occupy a page of their own, which is every layout but one: the cover that
  % carries the contents under the title, where the list is short and the
  % number reads better right after the entry.
  \ifidib@covertoc
    \titlecontents{#1}[#2]
      {\addvspace{2pt}\sffamily\normalsize\hypersetup{linkcolor=black}}
      {\textcolor{navy}{\thecontentslabel}\hspace{1em}}
      {}
      {\hspace{1.2em}\textcolor{grey}{\thecontentspage}}
      []%
  \else
    \titlecontents{#1}[#2]
      {\addvspace{2pt}\sffamily\normalsize\hypersetup{linkcolor=black}}
      {\textcolor{navy}{\thecontentslabel}\hspace{1em}}
      {}
      {\titlerule*[0.6pc]{\textcolor{grey}{.}}\textcolor{grey}{\thecontentspage}}
      []%
  \fi
}
\idib@tocformat{section}{0em}
\idib@tocformat{subsection}{1.4em}
% The list of tables uses the same line as a section entry, with the table
% number where the section number goes.
\titlecontents{table}[0em]
  {\addvspace{2pt}\sffamily\normalsize\hypersetup{linkcolor=black}}
  {\textcolor{navy}{\tablename~\thecontentslabel}\hspace{1.2em}}
  {}
  {\titlerule*[0.6pc]{\textcolor{grey}{.}}\textcolor{grey}{\thecontentspage}}
  []
\newcommand{\idib@compacttoc}{%
  \begingroup
    \setlength{\parskip}{0.32em}\setlength{\parindent}{0pt}%
    \@starttoc{toc}%
  \endgroup
}

% Title block.  With a cover, \maketitle is a titlepage: that kernel environment drops to one column for the page and restores two afterwards, so the cover is independent of the column setting.  Without a cover it is the kernel's \maketitle, which calls \@maketitle and spans it across both columns, so only \@maketitle is defined there.
\ifidib@cover

  %% @brief Heading of a front-matter page: the size of the cover, no number and no entry of its own in the contents.
  \colorlet{fronthead}{navy}
  \newcommand{\idib@fronthead}[1]{%
    % \noindent because the front matter runs after the titlepage, whose zeroed
    % \parindent died with its group; without it the heading starts indented.
    {\noindent\sffamily\fontsize{18}{22}\selectfont\bfseries\color{fronthead}%
     #1\par}%
    \vspace{1.2em}%
  }

  %% @brief The rule, the authors and the date that close the cover.  It is a command rather than inline text because the two covers put it in different places: the default one at the foot, the tocpage one directly under the title.
  \newcommand{\idib@coverdata}{%
    {\color{navy}\rule{\linewidth}{0.8pt}\par}%
    \vspace{0.55em}%
    \begin{minipage}[t]{0.62\linewidth}
      \idib@metalabel{Autores}%
      \sffamily\small\color{black}%
      \renewcommand{\idib@entry}[4]{%
        ##1 & \textcolor{grey}{##2} & \textcolor{grey}{##3} \\%
      }%
      \begin{tabular}[t]{@{}l@{\hspace{2em}}l@{\hspace{2em}}l@{}}
        \idib@authors
      \end{tabular}\par
      \renewcommand{\idib@entry}[4]{%
        \ifblank{##4}{}{{\footnotesize\color{grey}##1: ##4\par}}%
      }%
      \idib@authors
    \end{minipage}\hfill
    \begin{minipage}[t]{0.34\linewidth}
      \raggedleft
      \idib@metalabel{Fecha}%
      {\sffamily\normalsize\color{black}\@date\par}%
      \ifdefempty{\idib@profesores}{}{%
        \vspace{0.8em}%
        \idib@metalabel{Profesores}%
        {\sffamily\normalsize\color{black}\idib@profesores\par}%
      }%
    \end{minipage}%
  }

  %% @brief Full-page cover: institution and subject, title, contents, authors and date.  Replaces the kernel's \maketitle.
  %% @note  The titlepage environment numbers the following page 1 and handles the one-column switch, so neither is done by hand here.
  % The tinted ground and the gold title belong to the cover that carries the
  % contents, where the page is full and the colour organises it.  A cover that
  % holds only a title is quieter without them, so this variant drops both even
  % under color=true, and its two lines above the title grow: at the size they
  % take in a running title block they would read as a stray caption.
  %
  % The choice is made here and not inside \maketitle: set within the titlepage
  % environment it would be local to that group and would not reach the contents
  % pages, which are typeset after it.
  \ifidib@tocpage
    \colorlet{covertitle}{black}
    \colorlet{fronthead}{black}
    \colorlet{fronttype}{black}
    \renewcommand{\idib@frontsize}{\fontsize{11}{14}\selectfont}
    \renewcommand{\idib@typesize}{\fontsize{13}{16}\selectfont}
  \else
    \colorlet{covertitle}{gold}
  \fi

  \renewcommand{\maketitle}{%
    \begin{titlepage}%
    % geometry keeps its own record of the column setting, so the cover asks for one column explicitly; without it the page comes back split in two.
    \newgeometry{left=2.8cm,right=2.8cm,top=2.5cm,bottom=2.5cm,onecolumn}%
    \setlength{\parskip}{0pt}\setlength{\parindent}{0pt}%
    \ifidib@tocpage\else
      \begin{tikzpicture}[remember picture,overlay]
        \fill[cream]
          (current page.south west) rectangle (current page.north east);
      \end{tikzpicture}%
    \fi
    % The mark stands in for the institution's name here, and the running head carries it as text; both come from the same \institution field.
    {\sffamily\small\bfseries\color{grey}%
     \raisebox{-0.15em}{\includegraphics[height=0.32cm]{\idib@mark}}%
     \ifdefempty{\@subject}{}{%
       \enspace{\normalfont\color{grey}$\cdot$}\enspace
       {\normalfont\@subject}%
     }\par}%
    {\color{navy}\rule{\linewidth}{1.5pt}\par}%
    % With the contents on their own page the cover has nothing to fill it, so
    % the title is centred in what is left rather than left at the top with a
    % gap under it.  Without the option the layout is the original one.
    % Two arrangements of the same pieces.  With the contents on the cover the
    % page fills itself and the author block closes it at the foot.  Without
    % them there is nothing under the title, and a block left at the foot opens
    % a hole in the middle of the page; grouping title and authors in the upper
    % third puts the white space at the bottom, where it reads as margin.
    % Two arrangements of the same pieces.  With the contents on the cover the
    % title sits at the top and the list fills the page under it.  With the
    % contents moved out, the title is alone and is centred in the space that
    % is left, on a narrower measure so it breaks into more lines and reads as
    % a block rather than as a banner.
    \ifidib@tocpage
      \vfill
      \begin{center}
        \begin{minipage}{0.76\linewidth}
          \centering
          \idib@frontlines
          \vspace{0.6em}%
          {\sffamily\fontsize{28}{34}\selectfont\bfseries\color{covertitle}%
           \@title\par}%
        \end{minipage}
      \end{center}
      \vfill\vfill
    \else
      \vspace{2.6em}%
      \idib@frontlines
      {\sffamily\fontsize{28}{34}\selectfont\bfseries\color{gold}\@title\par}%
      \vspace{2.2em}%
      \idib@compacttoc
      \vfill
    \fi
    \idib@coverdata
    \restoregeometry
    \end{titlepage}%
    % The contents page is plain: no running head and no folio, so it reads as
    % front matter rather than as the first page of the body.
    \ifidib@tocpage
      \pagenumbering{roman}%
      \thispagestyle{empty}%
      \idib@fronthead{\contentsname}%
      \idib@compacttoc
      \clearpage
      \pagenumbering{arabic}\setcounter{page}{1}%
    \fi
  }

\else

  % The first page carries the logo alone, where the folio sits later: nothing on the left, so the top of the page is the logo and the title starts at the top of the text area instead of below a logo set in the flow.  \smash keeps the header box its normal height and lets the logo occupy the margin above it.
  \fancypagestyle{plain}{%
    \fancyhf{}%
    \fancyhead[R]{\smash{\raisebox{-0.72cm}{\includegraphics[width=2.05cm]{\idib@logo}}}}%
  }
  \renewcommand{\@maketitle}{%
    \setlength{\parindent}{0pt}\setlength{\parskip}{0pt}%
    % Spacing between the blocks of the front matter uses the kernel's own rubber skips (3, 6 and 12 pt) rather than hand-picked lengths, so it stretches under page-breaking pressure and follows the body size.
    \idib@frontlines
    {\sffamily\fontsize{22}{26}\selectfont\bfseries\color{black}\@title\par}%
    \smallskip
    {\color{navy}\rule{\linewidth}{0.8pt}\par}%
    \smallskip
    \begin{minipage}[t]{0.66\linewidth}
      \idib@metalabel{Autores}%
      {\sffamily\small\color{black}\idib@authorline@marked\par}%
      \smallskip
      {\sffamily\footnotesize\color{grey}\idib@authordetails}%
    \end{minipage}\hfill
    \begin{minipage}[t]{0.30\linewidth}
      \raggedleft
      \idib@metalabel{Fecha}%
      {\sffamily\small\color{black}\@date\par}%
    \end{minipage}%
    % The stripped abstract of the two-column layout brings its own separation above the box, which the one-column one has not; measured on the output that is about a \medskip, so one column adds one and both open a comparable gap before the summary.
    \par\if@twocolumn\smallskip\else\bigskip\fi
  }
\fi

%% @brief Declare the keywords, in the preamble.  They are typeset under the abstract; leave the command out and none are printed.
%% @note  They ride inside the abstract's own block rather than a second one, because two stripped blocks in a row leave a stray glyph between them.
\newcommand{\@keywords}{}
\newcommand{\keywords}[1]{\renewcommand{\@keywords}{#1}}

% Abstract: a filled box like the theory boxes of idib-academics.  Both the tint and the bar are navy, which the monochrome layout has already turned to black, so the same definition gives a blue box in colour and a grey one in mono.
\tcbset{
  idib@abstractbox/.style={
    enhanced, breakable,
    before skip=8pt plus 2pt minus 2pt,
    after skip=8pt plus 2pt minus 2pt,
    left=12pt, right=10pt, top=4pt, bottom=6pt,
    colback=navy!7, colframe=navy!7, boxrule=0pt, arc=0pt,
    borderline west={2.5pt}{0pt}{navy},
    before upper={%
      \setlength{\parskip}{0pt}\setlength{\parindent}{0pt}%
      {\sffamily\small\bfseries\color{navy}RESUMEN\par}\vspace{1pt}%
      \small\upshape\mdseries\rmfamily\noindent
    },
    parbox=false,
  },
}
%% @brief Summary box headed "RESUMEN" (redefines article's abstract).
%% @note  In two columns it is stripped across both, so the summary always reads as a single full-width block.
\renewenvironment{abstract}{%
  \if@twocolumn\begin{strip}\fi
  \begin{tcolorbox}[idib@abstractbox]%
}{%
  \end{tcolorbox}%
  \ifdefempty{\@keywords}{}{%
    \par\smallskip\noindent
    {\sffamily\small\bfseries Palabras clave:\enspace}{\small\@keywords}\par
  }%
  % Two columns need the extra air to separate the stripped block from the text
  % that flows around it; one column does not.  They need more than it looks:
  % after the strip the text resumes at the top of a column, where TeX discards
  % the following section's own pre-skip, while one column keeps it — measured,
  % the same gap comes out 16.8 pt against 22.1 pt.  The medskip makes up that
  % difference so both layouts open the same gap before the text.
  % Without a cover the summary and its keywords are the whole front matter, so
  % the group gets a little more room on both sides than it does under a cover,
  % where the cover already framed it.
  \if@twocolumn\bigskip\medskip\fi
  \ifidib@cover\else\smallskip\fi
  \if@twocolumn\end{strip}\fi
}

% Sections (CUP-style spacing)
\RequirePackage{needspace}
\RequirePackage{placeins}
% A heading with less than four lines of room under it moves to the next page
% with its text.  The kernel only discourages the break with \@secpenalty, and
% a penalty is a preference that TeX overrides when the alternative is worse.
%\newcommand{\sectionbreak}{\FloatBarrier\Needspace*{4\baselineskip}}
%\newcommand{\subsectionbreak}{\FloatBarrier\Needspace*{3\baselineskip}}
\newcommand{\sectionbreak}{}
\newcommand{\subsectionbreak}{}

\titleformat{\section}[block]
  {\sffamily\bfseries\normalsize\color{black}}
  {\normalsize\color{navy}\thesection\enspace}
  {0pt}{}
\titlespacing*{\section}
  {0pt}{2.2ex plus .5ex minus .2ex}{0.2ex}
\titleformat{\subsection}[block]
  {\sffamily\mdseries\itshape\normalsize\color{black}}
  {\normalsize\mdseries\color{navy}\upshape\thesubsection\enspace}
  {0pt}{}
\titlespacing*{\subsection}
  {0pt}{2ex plus .5ex minus .2ex}{0.2ex}

%% @brief Unnumbered section that still appears in the table of contents.
%% @note  Appendices use the kernel's \appendix followed by ordinary \section.
\newcommand{\freesection}[1]{%
  \section*{#1}\addcontentsline{toc}{section}{#1}%
}

% Without the wide margin there is nowhere to put a margin note, so the two commands become no-ops rather than errors.
\ifidib@sidemargin\else
  \renewcommand{\sidenote}[2][]{}
  \renewcommand{\sideimage}[3][]{}
\fi

% Headers and footers: institution and subject on odd pages, authors on even.
\pagestyle{fancy}
\fancyhf{}
\ifidib@sidemargin
  \fancyhfoffset[R]{\dimexpr\marginparwidth+\marginparsep\relax}
\fi
\fancyhead[L]{\idib@head@left}
\fancyhead[R]{\idib@head@right}

% Bibliography setup fixed for natbib
\RequirePackage[numbers,sort&compress]{natbib}
\renewcommand{\bibsection}{\section*{\refname}}

% hyperref (must be last)
\RequirePackage[
  unicode, colorlinks=true,
  linkcolor=navy, urlcolor=navy, citecolor=navy,
  pdfborder={0 0 0}, bookmarks=true, bookmarksopen=false
]{hyperref}

\endinput
%% End of file `idib-article.cls'.